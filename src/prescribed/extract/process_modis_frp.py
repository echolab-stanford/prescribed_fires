import os
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
    confidence_min: float = 30,
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
    confidence_min : float
        Confidence threshold to filter the data. Default is 50. The MODIS data comes with  confidence values from 0 to 100 to describe the level of certainty of the fire detection.
        See more here: https://modis-fire.umd.edu/files/MODIS_C61_BA_User_Guide_1.1.pdf.

        The defaul is 30 as it is the minimum value to consider a fire detection with nominal confidence: https://modis-fire.umd.edu/files/MODIS_C6_Fire_User_Guide_C.pdf

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

    # Filter by confidence
    df_points_proj = df_points_proj[df_points_proj["confidence"] >= confidence_min]

    year_files = []
    for year in tqdm(
        df_points_proj.timestamp.dt.year.unique(), desc="Array-ing the data"
    ):
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

        # add confidence as an attribute to the file
        frp_year = frp_year.rename({"x": "lon", "y": "lat"})

        df = frp_year.drop_vars(["spatial_ref"]).to_dataframe().reset_index().dropna()
        df["year"] = year
        year_files.append(df)

    # Open template file
    template_expanded = prepare_template(template_path)

    # Concatenate all the data and merge with the template
    concat_data_raw = pd.concat(year_files)
    concat_data = concat_data_raw.merge(template_expanded, on=["lat", "lon", "year"])

    # Drop lat and lon columns temporarily to avoid double columns
    concat_data = concat_data.drop(columns=["lat", "lon"])

    # Add count of fires per year in a cummulative way
    concat_data["fire"] = 1
    concat_data["confidence_min"] = confidence_min
    concat_data = concat_data.merge(
        template_expanded, on=["grid_id", "year"], how="right"
    )

    # Calculate the count of fires and the cumulative sum of FRP to pass
    # to the balancing function
    concat_data["count_fires"] = concat_data.groupby(["grid_id"]).fire.transform(
        "cumsum"
    )

    concat_data["cummax_frp"] = concat_data.groupby(["grid_id"]).frp.transform("cummax")

    # Fill NaNs with the laterst value (ffil)
    concat_data["count_fires"] = concat_data.groupby(
        "grid_id", as_index=False
    ).count_fires.ffill()

    concat_data["cummax_frp"] = concat_data.groupby(
        "grid_id", as_index=False
    ).cummax_frp.ffill()

    # Fill NaNs with 0 and assume no frp is no fire
    concat_data["count_fires"].fillna(0, inplace=True)
    concat_data["cummax_frp"].fillna(0, inplace=True)
    concat_data["frp"].fillna(0, inplace=True)

    if wide:
        # Pivot the table to wide format and save to feather
        wide_frp = pd.pivot(
            concat_data,
            index="grid_id",
            columns="year",
            values=["count_fires", "cummax_frp"],
        )
        wide_frp.columns = [f"{col}_{idx}" for col, idx in wide_frp.columns]
        wide_frp = wide_frp.reset_index()

        log.info("Saving PRISM in wide format")
        wide_frp.to_feather(os.path.join(save_path, "frp_wide.feather"))

    # Save to feather
    log.info("Saving PRISM in long format")
    concat_data.to_feather(os.path.join(save_path, "frp_concat.feather"))

    return None
