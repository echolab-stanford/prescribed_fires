"""Use MTBS data to create a treatment identification based on a template

This function takes a MTBS shapefile and rasterizes it using a user defined
template. The function will spit out a NetCDF file with the rasterized
polygons using the "Event_ID" columns as the values of the data. To do this,
we only use the numeric part of the event ID. Thus:

Event: CA3607412018819840329 will be converted to 3607412018819840329

Each year will be a dimension in the NetCDF file.
"""

import logging
from functools import partial, reduce
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from prescribed.utils import calculate_fire_pop_dens, prepare_template
from xrspatial import proximity

log = logging.getLogger(__name__)


def transform_to_category(
    data: xr.DataArray, var_name: str, dummy: bool = False
) -> pd.DataFrame:
    """Util to transform xarray to df and add category values"""

    data_ = data.copy()

    if not dummy:
        # Take from idx in events enums to actual event_id names as strings
        event_id_str = data_["Event_ID_categories"][
            data_["Event_ID"].astype(int)
        ].drop_vars("Event_ID_categories")
        data_["Event_ID"] = event_id_str
    else:
        # Create dummy variables with 1 to everything that has data.
        data_ = xr.where(data_ == -1, np.nan, data_)
        data_ = xr.where(data_ != np.nan, 1, data_)

    # Rename xarray dimensions: x -> lon, y -> lat
    data_ = data_.rename({"x": "lon", "y": "lat"})

    # Convert xarray to dataframe and add correct event_id data back!
    data_df = (
        data_.drop_dims(["Event_ID_categories"])
        .to_dataframe()
        .reset_index()
        .dropna()
        .drop(columns=["spatial_ref"], errors="ignore")
        .rename(columns={"Event_ID": var_name})
    )

    return data_df


def create_distances(
    mtbs_shapefile: Union[gpd.GeoDataFrame, str],
    template: Union[xr.DataArray, str],
    save_path: Optional[str] = None,
    pop_threshold: Optional[float] = 0.5,
    buffer_treatment: Optional[int] = 5_000,
    **kwargs,
) -> None:
    """Create gridded treatments from MTBS dataset using a grid template

    This function takes a MTBS shapefile and rasterizes it using a user defined
    template. The function will spit out a Feather tabular file with the rasterized
    polygons using the "Event_ID" columns as the values of the data. Additionally,
    the file will contain a selection of the columns from the original MTBS:
        - Ig_Date
        - Incid_Name
        - Incid_Type

    If the spillovers option is passed, the function will treat the buffer around
    polygons as a treatment only. This is using a utility function in the library
    (see prescribed.utils.calculate_fire_pop_dens), and you can pass additional
    option using the kwargs parameter. The user must define a threshold to filter
    the polygons with the highest population density according to a percentile.
    The default is to use the median.

    Sources:
    -https://www.mtbs.gov/direct-download

    Parameters
    ----------
    mtbs_shapefile : str
        Path to the MTBS shapefile to process
    template : str
        Path to the template file to reproject to
    save_path : str
        Path to save the resulting output
    pop_threshold : Optional[float], optional
        The percentile to filter the population density, by default 0.5
    buffer_treatment : Optional[int], optional
        The buffer distance in meters for the treatment definition, by default
        10000
    kwargs : dict
        Additional parameters for the `calculate_fire_pop_dens` function

    Returns
    -------
    pd.DataFrame
        Saves a Feather file to the save_path and returns a DataFrame with the
        rasterized data
    """

    # Read in data frame and filter for California

    if isinstance(mtbs_shapefile, str):
        mtbs = gpd.read_file(mtbs_shapefile)
        mtbs["Ig_Date"] = pd.to_datetime(mtbs["Ig_Date"])

        # Filter for wildfires and prescribed fires in California
        wildfires = mtbs[
            (mtbs.Event_ID.str.contains("CA"))
            & (mtbs.Incid_Type.isin(["Wildfire", "Prescribed Fire"]))
        ]
    else:
        wildfires = mtbs_shapefile

    # Open template as xarray
    if isinstance(template, str):
        template = rioxarray.open_rasterio(template)

    # Reproject to same proj as template (CA Albers)
    if wildfires.crs.to_epsg() != template.rio.crs.to_epsg():
        wildfires = wildfires.to_crs(4326)
        wildfires_meters = wildfires.to_crs(template.rio.crs.to_epsg())
    else:
        wildfires_meters = wildfires.copy()

    # Sample polygons by population ("remoteness")
    if pop_threshold is not None:
        pop_dens = calculate_fire_pop_dens(
            geoms=wildfires,
            pop_raster_path=kwargs["pop_raster_path"],
            buffer=kwargs.get("buffer", 10_000),
            date_col=kwargs.get("date_col", "Ig_Date"),
            mask=kwargs.get("mask", None),
            template=template,
        )

        # Calculate population threshold and subset geometry
        tresh = pop_dens.total_pop.quantile([pop_threshold]).values[0]
        wildfires_meters = pop_dens[pop_dens["total_pop"] <= tresh]

        # Add treshold and buffer distance
        wildfires_meters = wildfires_meters.assign(
            tresh=tresh, boundary=wildfires_meters.geometry.boundary
        )

        # Drop unnecesary geometry columns from population subset
        wildfires_meters = wildfires_meters.drop(columns=["donut"], errors="ignore")
    else:
        wildfires_meters = wildfires_meters.assign(
            tresh=np.nan, boundary=wildfires_meters.geometry.boundary
        )

    # Create year column to add time dimension
    wildfires_meters["year"] = wildfires_meters.Ig_Date.dt.year

    # and now create the geoms we need to define treatment/control areas
    wildfires_meters["buffer"] = wildfires_meters.geometry.buffer(buffer_treatment)
    wildfires_meters["donut"] = wildfires_meters["buffer"].difference(
        wildfires_meters["geometry"]
    )

    # Loop through each year and rasterize the polygons. We cannot execute this
    # operation in all the years, since we want to calculate the distance to the
    # closest treatment for each pixel in the year of the fire, so we do not have
    # cases where we pick a non-contemporaneous treatment in space.
    data_list_wildfire = []
    for _, data in wildfires_meters.groupby("year"):
        # Create a enums dict with category values
        enums = {"Event_ID": data["Event_ID"].unique().tolist()}

        # Use geocube to rasterize the shapefile polygons (real fire polygons)
        data = data.set_geometry("geometry", crs=data.crs)
        wildfire_polygons = make_geocube(
            vector_data=data,
            measurements=["Event_ID"],
            like=template,
            categorical_enums=enums,
            rasterize_function=partial(rasterize_image, all_touched=True),
        )

        df_wildfire = transform_to_category(wildfire_polygons, var_name="wildfire")

        # Use geocube to rasterize the fire boundary
        data["geometry"] = (
            data.geometry.boundary.values
        )  # gpd.set_geometry() not working

        wildfire_boundaries = make_geocube(
            vector_data=data,
            measurements=["Event_ID"],
            like=template,
            categorical_enums=enums,
            rasterize_function=partial(rasterize_image, all_touched=True),
        )

        df_boundaries = transform_to_category(wildfire_boundaries, var_name="boundary")

        # Use geocube to rasterize the shapefile donuts (buffer around fires)
        data["geometry"] = data["donut"].values  # gpd.set_geometry() not working
        wildfire_polygons = make_geocube(
            vector_data=data,
            measurements=["Event_ID"],
            like=template,
            categorical_enums=enums,
            rasterize_function=partial(rasterize_image, all_touched=True),
        )

        df_donut = transform_to_category(wildfire_polygons, var_name="donut")

        # Calculate distance (proximity) to each event in loop year
        wildfire_boundaries = xr.where(
            wildfire_boundaries == -1, np.nan, wildfire_boundaries
        )

        dist_arr = proximity(
            wildfire_boundaries.Event_ID,
            max_distance=buffer_treatment,
            distance_metric="EUCLIDEAN",
        )

        # Rename xarray dimensions: x -> lon, y -> lat
        dist_arr = dist_arr.rename({"x": "lon", "y": "lat"})

        # Add proximity to the dataframe
        df_dist = (
            dist_arr.to_dataframe(name="distance")
            .reset_index()
            .drop(columns=["spatial_ref"], errors="ignore")
            .dropna()
        )

        # Merge all data frames on lat/lon with reduce and add the year
        merged = reduce(
            lambda left, right: pd.merge(left, right, on=["lat", "lon"], how="outer"),
            [df_wildfire, df_boundaries, df_donut, df_dist],
        )
        merged["year"] = data.year.unique()[0]

        # Remove the wildfire polygons from the distances
        merged = merged[merged["wildfire"] == "nodata"]

        # Append to list
        data_list_wildfire.append(merged)

    # Concatenate all data frames
    data_wildfire = pd.concat(data_list_wildfire)

    # Merge with template to get grid_id
    template_df = (
        prepare_template(template)
        .groupby("grid_id", as_index=False)
        .first()
        .reset_index(drop=True)
        .drop(columns=["year"], errors="ignore")
    )
    data_wildfire = data_wildfire.merge(template_df, on=["lat", "lon"], how="left")

    # Just return the right stuff (the buffer/donut distances)
    data_wildfire = data_wildfire[
        [
            "grid_id",
            "lat",
            "lon",
            "year",
            "donut",
            "wildfire",
            "distance",
        ]
    ]

    return data_wildfire
