import os
import pandas as pd
import geopandas as gpd
import xarray as xr

from tqdm import tqdm
from geopandas.tools import sjoin
from geocube.api.core import make_geocube

from src.utils import prepare_template


def process_modis_file(file_path, save_path, aoi, template, feather=False, wide=False):
    """Process MODIS file from FIRMS and transform into array format

    This function takes a CSV file from FIRMS and transforms it into an xarray
    object with the correct gridding over a defined resolution and region. We
    use geocube gridding routines to create a file for each year. Each yearly
    array includes data for each day combining all measurements by both sensors
    in the MODIS constellation (i.e. terra and aqua).

    Parameters
    ----------
    file_path : str
        Path to the CSV file to process
    save_path : str
        Path to save the resulting output
    aoi : str
        Path to the crosswalk file (shapefile)
    template : str
        Path to the template file to reproject to
    feather : bool
        If True, save all data as a single feather file in long format, unless wide is True. In that case it will save both file in feather format.
    wide : bool
        If True, save the data in wide format with a cumulative sum and count for count the cumulative fire behavior. It will save both wide and long files in feather format.

    Returns
    -------
    None
        Saves yearly NetCDF files with the processed data or a single feather file in either long or wide format.
    """

    # Read the CSV file
    df = pd.read_csv(file_path)
    aoi = gpd.read_file(aoi)

    # Create date column
    df["acq_date"] = pd.to_datetime(df.acq_date)

    # Transform CSV to geopandas dataframe
    df_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # Read the template file with xarray
    template = xr.open_dataarray(template)

    # Transform things to meters CA to avoid problems
    df_points_proj = df_points.to_crs(aoi.crs.to_epsg())

    # Spatial join to filter all points out of the aoi
    df_points = sjoin(df_points_proj, aoi, how="inner")

    # Group per each year and create yearly arrays
    groupped_date = df_points_proj.groupby(df_points_proj["acq_date"].dt.year)

    feather_files = []
    for year, group in tqdm(groupped_date, desc="Array-ing the data"):
        arrays_date = []
        for name, group_day in group.groupby(["acq_date"]):
            geo_grid = make_geocube(
                vector_data=group_day,
                measurements=["frp"],
                like=template,
            )
            geo_grid = geo_grid.expand_dims({"time": name})
            arrays_date.append(geo_grid)

        # Concat daily arrays
        frp_year = xr.concat(arrays_date, dim="time")
        frp_year = frp_year.sortby("time").rename({"x": "lon", "y": "lat"})

        # Create save path if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_stem = f"frp_modis_firms_{int(year)}"
        path_to_save = os.path.join(save_path, f"{file_stem}.nc4")
        frp_year.to_netcdf(path_to_save)

        if feather:
            df = (
                frp_year.drop_vars(["spatial_ref"])
                .to_dataframe()
                .reset_index()
                .dropna()
            )
            df["year"] = df["time"].dt.year
            feather_files.append(df)

    # Save feather files if needed
    # Open template file
    template_expanded = prepare_template(template)

    if feather:
        concat_data = pd.concat(feather_files)
        concat_data = concat_data.merge(template_expanded, on=["lat", "lon", "year"])

        if wide:
            concat_data = concat_data.groupby(
                ["grid_id", "year"], as_index=False
            ).frp.max()

            # Add count of fires per year in a cummulative way
            concat_data["fire"] = 1
            concat_data["count_fires"] = concat_data.groupby(["grid_id"]).fire.cumsum()
            concat_data = concat_data.merge(
                template_expanded, on=["grid_id", "year"], how="right"
            )

            # Add fire aggregations and cumsum to get the cumulative frp for all the time
            concat_data["cum_frp"] = concat_data.groupby(
                ["grid_id"], as_index=False
            ).frp.cummax()

            # Forward fill all NAs after merge and fill the rest with 0
            concat_data.update(concat_data.groupby(["grid_id"]).ffill().fillna(0))

            # Pivot the table to wide format and save to feather
            wide_frp = pd.pivot(
                concat_data,
                index="grid_id",
                columns="year",
                values=["cum_frp", "count_fires"],
            )
            wide_frp.columns = [f"{col}_{idx}" for col, idx in wide_frp.columns]
            wide_frp = wide_frp.reset_index()

            wide_frp.to_feather(os.path.join(save_path, "frp_wide.feather"))

    # Save to feather
    concat_data.to_feather(os.path.join(save_path, "frp_concat.feather"))

    return None
