import os
import pandas as pd
import geopandas as gpd
import xarray as xr

from tqdm import tqdm
from geopandas.tools import sjoin
from geocube.api.core import make_geocube


def process_modis_file(file_path, save_path, aoi, template):
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

    Returns
    -------
    None
        Saves yearly NetCDF files with the processed data
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

        # Save yearly array

        # Create save path if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        path_to_save = os.path.join(save_path, f"frp_modis_firms_{int(year)}.nc4")
        frp_year.to_netcdf(path_to_save)

    return None
