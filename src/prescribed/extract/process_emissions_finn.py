import os
import logging
from pathlib import Path
from typing import List, Optional, Union

import duckdb
import geopandas as gpd
import pandas as pd
import rioxarray
import xarray as xr
from geocube.api.core import make_geocube
from tqdm import tqdm

from ..utils import prepare_template

logger = logging.getLogger(__name__)


def process_finn(
    files_path: Union[str, list],
    template: Union[str, xr.DataArray],
    aoi: Union[str, gpd.GeoDataFrame],
    save_path: str,
    save_agg: Optional[bool] = False,
    inventory: str = "GEOSCHEM",
    species: Optional[str] = None,
) -> Union[pd.DataFrame, xr.Dataset]:
    """Process emissions data from FINN

    This function processes emissions data from NCAR Fire Emissions Inventory

    Parameters
    ----------
    files_path : list
        List of paths to the files to process
    save_path : str
        Path to save the resulting output
    save_agg : bool
        If True, the function will save the aggregated emissions to a file. Default is False
    aoi : str or gpd.GeoDataFrame
        Path to the crosswalk file (shapefile) or a GeoDataFrame
    template : str
        Path to the template file to reproject to
    inventory : str
        Inventory to process. Default is GEOSCHEM if a list of files is provided. Otherwise, the function won't check if the files correspond to a specific inventory.
    species : list
        List of species to process. Default is None, which processes all species

    Returns
    -------
    pd.DataFrame or xr.Dataset
        If save_agg is True, the function will return a DataFrame with the aggregated emissions. Otherwise, it will return a xarray Dataset with the emissions data.
    """

    all_species: List[str] = [
        "CO2",
        "CO",
        "CH4",
        "NMOC",
        "H2",
        "NOXasNO",
        "SO2",
        "PM25",
        "TPM",
        "TPC",
        "OC",
        "BC",
        "NH3",
        "NO",
        "NO2",
        "NMHC",
        "PM10",
        "ACET",
        "ALD2",
        "ALK4",
        "BENZ",
        "C2H2",
        "C2H4",
        "C2H6",
        "C3H8",
        "CH2O",
        "GLYC",
        "GLYX",
        "HAC",
        "MEK",
        "MGLY",
        "PRPE",
        "TOLU",
        "XYLE",
        "NO_1",
    ]

    if not isinstance(files_path, list):
        file_list = list(Path(files_path).glob(f"*{inventory}*"))
    else:
        file_list = files_path

    if isinstance(template, str):
        template = rioxarray.open_rasterio(template)

    if isinstance(aoi, str):
        aoi = gpd.read_file(aoi)

    # Store in mem if no saving
    if not save_path:
        list_files = []

    for file in tqdm(file_list, desc=f"Processing the {inventory}", position=0):
        # Get year from file name
        # This is relying in NCAR's naming convention. For example:
        # FINNv2.5_modvrs_{inventory}_{year}_c20211213.txt.gz
        year: int = file.stem.split("_")[3]

        if not os.path.exists(os.path.join(save_path, f"finn_{inventory}_{year}.nc")):
            # Why pandas when you have duckdb? :D (I'm kidding, I love pandas)
            data: pd.DataFrame = duckdb.query(
                f"SELECT * FROM read_csv('{file}', header = true, auto_detect=true)"
            ).to_df()

            # Get correct year dates
            data["date"] = data["DAY"].apply(lambda x: f"{year}-{x}")
            data["date"] = pd.to_datetime(data["date"], format="%Y-%j")

            # Create geodataframe
            emissions: gpd.GeoDataFrame = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data["LONGI"], data["LATI"]),
                crs="EPSG:4326",
            )

            # Reproject data to the aoi crs
            emissions = emissions.to_crs(aoi.crs)

            # Subset data to only include points in the AOI
            emissions = gpd.sjoin(emissions, aoi, how="inner")

            # Create cube by day
            emissions_doy_arr = []
            for doy in tqdm(
                emissions["date"].unique(),
                desc=f"Processing days [year: {year}]",
                position=1,
            ):
                cube: xr.DataArray = make_geocube(
                    vector_data=emissions[emissions["date"] == doy],
                    measurements=all_species if species is None else species,
                    like=template,
                )
                cube = cube.expand_dims({"time": [doy]})
                emissions_doy_arr.append(cube)

            # Concatenate all the arrays by day
            cube: xr.Dataset = xr.concat(emissions_doy_arr, dim="time")

            if save_path:
                # Create save path if it does not exist
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                # Save yearly NetCDF file
                cube.to_netcdf(os.path.join(save_path, f"finn_{inventory}_{year}.nc"))
            else:
                return list_files.append(cube)
        else:
            logging.info(f"File for year {year} already exists. Skipping.")

    if save_agg:
        files = list(Path(save_path).glob("*.nc"))

        # Create aggregated emissions
        xarr = xr.open_mfdataset(files)

        # If using aerosols, data is aggregated at the kg/day level. Thus with the mean along time we are taking the sum of the year by day
        agg = xarr.mean(dim="time")
        agg_df = agg.to_dataframe().reset_index().drop(columns=["spatial_ref"]).dropna()

        try:
            template_df = prepare_template(template).groupby("grid_id").first()
        except ValueError as e:
            raise ValueError(
                (
                    "You need to pass a path to the template as we are expecting that for the",
                    f"aggregation. [{e}]",
                )
            )

        # Merge template with the aggregated data
        agg_df = template_df.merge(agg_df, on="grid_id")
        agg_df.to_feather(os.path.join(save_path, f"finn_{inventory}_agg.feater"))

        return agg_df

    else:
        cube = xr.concat(list_files, dim="time")

    return cube


def aggregate_finn_fires(
    path_files: Union[str, list],
    fire_shape: Union[gpd.GeoDataFrame, str],
    save_path: str,
    agg_func: Optional[str] = "sum",
    start_date_var: Optional[str] = None,
    end_year_var: Optional[str] = None,
) -> None:
    """Aggregate FINN emissions data to specific fire shapefile and aggregate data.

    This function takes daily emission inventories from FINN and processes them to be aggregated to a particular fire shapefile.
    The FINN data includes fire polygon identifiers (POLYID and FIREID), which are derived from pixel proximity from the MODIS raw data and not from a particular fire polygon dataset.

    This function spatially merges all points to respective fires in the user-defined fire shape and then aggregates them from the start date to the end of the year.

    Parameters
    ----------
    path_files : Union[str, list]
        The path or list of paths to the FINN emission files.
    fire_shape : Union[gpd.GeoDataFrame, str]
        The fire shapefile or path to the fire shapefile to which the emissions will be aggregated.
    agg_func : Optional[str], default="sum"
        The aggregation function to use when aggregating the emissions. Options include "sum", "mean", "max", "min", etc.
    start_date_var : Optional[str], default=None
        The variable name in the FINN data that represents the start date of the emissions. If None, the function will use the first available date.
    end_year_var : Optional[str], default=None
        The variable name in the FINN data that represents the end year of the emissions. If None, the function will use the last available year.

    Returns
    -------
    None
        This function does not return anything. It saves the aggregated emissions to a file.

    """
    # Read the fire shapefile if it is a path
    if isinstance(fire_shape, str):
        fire_shape = gpd.read_file(fire_shape)
