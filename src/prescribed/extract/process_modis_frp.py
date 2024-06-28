import os
import pdb
import logging
import pandas as pd
import geopandas as gpd
import xarray as xr

from tqdm import tqdm
from geopandas.tools import sjoin
from geocube.api.core import make_geocube

from prescribed.utils import prepare_template

log = logging.getLogger(__name__)


def process_modis_file(
    file_path,
    save_path,
    aoi,
    template_path,
    wide=False,
    confidence: float = 50.0,
):
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
    template_path : str
        Path to the template file to reproject to
    wide : bool
        If True, save the data in wide format with a cumulative sum and count for count the cumulative fire behavior. It will save both wide and long files in feather format.
    confidence : float
        Confidence threshold to filter the data. Default is 50. The MODIS data comes with  confidence values from 0 to 100 to describe the level of certainty of the fire detection.
        See more here: https://modis-fire.umd.edu/files/MODIS_C61_BA_User_Guide_1.1.pdf

    Returns
    -------
    None
        Saves yearly NetCDF files with the processed data or a single feather file in either long or wide format.
    """

    # Create save dir if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Read the CSV file to start processing
    df = pd.read_csv(file_path)
    aoi = gpd.read_file(aoi)

    # Create date column using the date and time
    df["timestamp"] = (
        df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4)
    )  # This is a small trick to make it HH:MM
    df["timestamp"] = pd.to_datetime(df.timestamp)

    # Subset to keep the good pixels with confidence
    df = df[df["confidence"] >= confidence]

    # Transform CSV to geopandas dataframe
    df_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )

    # Read the template file with xarray
    template = xr.open_dataarray(template_path)

    # Transform things to meters CA to avoid problems
    df_points_proj = df_points.to_crs(aoi.crs.to_epsg())

    # Spatial join to filter all points out of the aoi
    df_points_proj = sjoin(df_points_proj, aoi, how="inner")

    year_files = []
    for year in tqdm(
        df_points_proj.timestamp.dt.year.unique(), desc="Array-ing the data"
    ):
        # Check if file exists, otherwise skip and load from storage
        file_stem = f"frp_modis_firms_{int(year)}"
        path_to_save = os.path.join(save_path, f"{file_stem}.nc4")

        if not os.path.exists(path_to_save):
            year_df = df_points_proj[df_points_proj.timestamp.dt.year == year]
            # Take the max value of the FRP for each pixel-year
            year_df = year_df.groupby(
                ["latitude", "longitude", "geometry"], as_index=False
            ).frp.max()

            frp_year = make_geocube(
                vector_data=year_df,
                measurements=["frp"],
                like=template,
            )

            # Save file to avoid re-processing when building other stuff.
            frp_year.attrs["confidence"] = confidence
            frp_year = frp_year.rename({"x": "lon", "y": "lat"})
            frp_year.to_netcdf(path_to_save)
        else:
            frp_year = xr.open_dataset(path_to_save)

        df = frp_year.drop_vars(["spatial_ref"]).to_dataframe().reset_index().dropna()
        df["year"] = year
        year_files.append(df)

    # Open template file
    template_expanded = prepare_template(template_path)

    # Concatenate all the data and merge with the template
    concat_data = pd.concat(year_files)
    concat_data = concat_data.merge(template_expanded, on=["lat", "lon", "year"])

    # Drop lat and lon columns temporarily to avoid double columns
    concat_data = concat_data.drop(columns=["lat", "lon"])

    # Add count of fires per year in a cummulative way
    concat_data["fire"] = 1
    concat_data["count_fires"] = concat_data.groupby(["grid_id"]).fire.cumsum()
    concat_data = concat_data.merge(
        template_expanded, on=["grid_id", "year"], how="right"
    )

    # Forward fill all NAs after merge and fill the rest with 0
    concat_data.update(concat_data.groupby(["grid_id"]).ffill().fillna(0))

    if wide:
        # Pivot the table to wide format and save to feather
        wide_frp = pd.pivot(
            concat_data,
            index="grid_id",
            columns="year",
            values=["count_fires"],
        )
        wide_frp.columns = [f"{col}_{idx}" for col, idx in wide_frp.columns]
        wide_frp = wide_frp.reset_index()

        log.info("Saving PRISM in wide format")
        wide_frp.to_feather(os.path.join(save_path, "frp_wide.feather"))

    # Save to feather
    log.info("Saving PRISM in long format")
    concat_data.to_feather(os.path.join(save_path, "frp_concat.feather"))

    return None
