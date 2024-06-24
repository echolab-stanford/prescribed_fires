"""Use MTBS data to create a treatment identification based on a template

This function takes a MTBS shapefile and rasterizes it using a user defined
template. The function will spit out a NetCDF file with the rasterized
polygons using the "Event_ID" columns as the values of the data. To do this,
we only use the numeric part of the event ID. Thus:

Event: CA3607412018819840329 will be converted to 3607412018819840329

Each year will be a dimension in the NetCDF file.
"""

import logging
import os
from functools import partial
from typing import Optional, Union
from pathlib import Path

import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
import rioxarray
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image
from omegaconf import DictConfig
from prescribed.utils import calculate_fire_pop_dens

log = logging.getLogger(__name__)


def create_treatments(
    mtbs_shapefile: Union[gpd.GeoDataFrame, str],
    template: str,
    save_path: str,
    spillovers: bool = False,
    threshold: Optional[float] = 0.5,
    buffer_treatment: Optional[int] = 1000,
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
    spillovers : bool, optional
        If True, calculate population density for spillovers, by default False
    threshold : Optional[float], optional
        The percentile to filter the population density, by default 0.5
    buffer_treatment : Optional[int], optional
        The buffer distance in meters for the treatment definition, by default
        1000
    kwargs : dict
        Additional parameters for the `calculate_fire_pop_dens` function

    Returns
    -------
    None
        Saves a Feather file to the save_path
    """

    # Cols of interest
    cols = ["Event_ID", "Ig_Date", "Incid_Name", "Incid_Type"]

    # Read in data frame and filter for California
    mtbs = gpd.read_file(mtbs_shapefile)
    mtbs["Ig_Date"] = pd.to_datetime(mtbs["Ig_Date"])

    # Open template as xarray
    template = rioxarray.open_rasterio(template)

    # Filter for wildfires and prescribed fires in California
    wildfires = mtbs[
        (mtbs.Event_ID.str.contains("CA"))
        & (mtbs.Incid_Type.isin(["Wildfire", "Prescribed Fire"]))
    ]

    # Reproject to same proj as template (CA Albers)
    if wildfires.crs.to_epsg() != template.rio.crs.to_epsg():
        wildfires = wildfires.to_crs(4326)
        wildfires_meters = wildfires.to_crs(template.rio.crs.to_epsg())

    # If spillovers, calculate population density
    if spillovers:
        # Redefine cols
        cols = cols + ["buffer", "tresh", "buffer_treat"]

        pop_dens = calculate_fire_pop_dens(geoms=wildfires, template=template, **kwargs)

        # Calculate threshold and subset geometry
        tresh = pop_dens.total_pop.quantile([threshold]).values[0]
        wildfires_meters = pop_dens[pop_dens["total_pop"] <= tresh]

        # Create buffer and new treatment donut
        wildfires_meters["geometry"] = wildfires_meters.buffer(
            buffer_treatment
        ).difference(wildfires_meters.geometry)

        # Get buffer from kwargs
        buffer_pop = kwargs.get("buffer", np.nan)

        # Return treshold and buffer distance
        wildfires_meters["tresh"] = tresh
        wildfires_meters["buffer"] = buffer_pop
        wildfires_meters["buffer_treat"] = buffer_treatment

        # Drop unnecesary geometry columns
        wildfires_meters = wildfires_meters.drop(columns=["donut"], errors="ignore")

    # Create year column to add time dimension
    wildfires_meters["year"] = wildfires_meters.Ig_Date.dt.year

    # Create a enums dict with category values
    enums = {"Event_ID": wildfires_meters["Event_ID"].unique().tolist()}

    # Use geocube to rasterize the shapefile
    wildfire_arr = make_geocube(
        vector_data=wildfires_meters,
        measurements=["Event_ID"],
        like=template,
        categorical_enums=enums,
        group_by="year",
        rasterize_function=partial(rasterize_image, all_touched=True),
    )

    # Transform indices into categories within the array
    event_id_str = wildfire_arr["Event_ID_categories"][
        wildfire_arr["Event_ID"].astype(int)
    ].drop_vars("Event_ID_categories")
    wildfire_arr["Event_ID"] = event_id_str

    # Rename xarray dimensions: x -> lon, y -> lat
    wildfire_arr = wildfire_arr.rename({"x": "lon", "y": "lat"})

    # Convert xarray to dataframe and add correct event_id data back!
    wildfire_df = (
        wildfire_arr.drop_dims(["Event_ID_categories"]).to_dataframe().reset_index()
    )

    # Merge with subset data from original MTBS
    wildfire_merge = wildfires_meters[cols].merge(
        wildfire_df, on="Event_ID", how="right"
    )

    # Save the file
    if not os.path.exists(Path(save_path).parent):
        os.makedirs(Path(save_path).parent, exist_ok=True)

    wildfire_merge.to_feather(save_path)

    return None


@hydra.main(config_path="../conf", config_name="treatments")
def main(cfg: DictConfig) -> None:
    log.info("Building dataset")
    create_treatments(
        mtbs_shapefile=cfg.treatments.mtbs_shapefile,
        template=cfg.treatments.template,
        save_path=cfg.treatments.save_path,
        spillovers=cfg.treatments.spillovers,
        threshold=cfg.treatments.threshold,
        buffer_treatment=cfg.treatments.buffer_treatment,
        buffer=cfg.treatments.buffer,
        pop_raster_path=cfg.treatments.pop_raster_path,
        mask=cfg.treatments.mask,
    )

    log.info(f"File saved to {cfg.treatments.save_path}!")


if __name__ == "__main__":
    main()
