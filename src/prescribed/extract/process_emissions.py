import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
from odc.algo import xr_reproject
from tqdm import tqdm

from prescribed.utils import prepare_template

log = logging.getLogger(__name__)


def process_emissions_feather(
    list_files: List[Union[Path, pd.DataFrame]],
    template_path: str,
    save_path: str,
    extract_band: str = "PM2.5",
) -> None:
    """Aux file to process list of NetCDFs to feather file

    This function takes a file of emissions files in NetCDF format or DataFrames and reads them into
    feather format while adding the event_id in the process. The function will either
    take a list of files or a list of open objects in feather format.

    Parameters
    ----------
    list_files : list
        List of files to process. The files can be either a string with paths or a set of pd.Dataframe objects.
    template_path : str
        Path to the template file to reproject to. The template should be a GeoTIFF
    extract_band : str
        Band to extract from the NetCDF file. Following Xu (2022) data, this files have the following possible bands (13 in total):
         - evt_class
         - burn severity(cbi_severity)
         - emissions(CO2, CO, CH4, NMOC, SO2, NH3, NO, NO2, NOx, PM2.5, OC, BC)
    save_path : str
        Path to save the feather file. The file will be saved as 'emissions_long.feather'
    """

    # Check all elements in the list are a dataframe
    if all([isinstance(f, pd.DataFrame) for f in list_files[1:5]]):
        df = pd.concat(list_files)
    else:
        list_objs = []
        for p in tqdm(list_files, desc="Reading grd files into feather..."):
            ds = xr.open_dataarray(p)
            # .to_dataframe(extract_band).dropna().reset_index()
            if "band" in ds.coords and ds.coords["band"].size > 1:
                df = ds.isel(band=0).to_dataframe(extract_band).dropna().reset_index()
            else:
                df = ds.to_dataframe(extract_band).dropna().reset_index()

            df["event_id"] = p.stem
            list_objs.append(df)
        df = pd.concat(list_objs)

    # Merge with template
    template = (
        prepare_template(template_path).groupby("grid_id", as_index=False).first()
    )

    # Drop year column to avoid problems!
    if "year" in template.columns:
        template.drop(columns=["year"], inplace=True)

    # Remove weird_pixels (see analysis/weird_pixels.ipynb) by using an inner merge
    em_files = df.merge(template, on=["lat", "lon"])

    # If grid_id and year are floats, transform to int
    em_files["grid_id"] = em_files["grid_id"].astype(int)

    if "band" in em_files.columns:
        em_files.drop(columns=["band"], inplace=True)

    log.info(f"Saving file into {os.path.join(save_path, 'emissions_long.feather')}")
    em_files.to_feather(os.path.join(save_path, "emissions_long.feather"))

    return None


def process_emissions(
    emissions_path: str,
    template_path: str,
    save_path: str,
    extract_band: str,
    feather: bool = False,
    overwrite: bool = False,
):
    """Process emissions data for template (based on Xu (2022) datasets)

    Process emissions data (@ 30m resolution) to overlay with a user-defined template.
    The function expects separate GeoTIFF files for each event. The function will
    read each file and resample it to fit the template grid and projection. Each
    file will be saved with the same name as the original file in the save_path
    directory and the directory structure will be preserved.

    Parameters
    ----------
    emissions_path : str
        Path to the emission data. The data format should be GeoTIFF
    template_path : str
        Path to the template file to reproject to. The template should be a GeoTIFF
        file with the desired projection and grid
    save_path : str
        Path to save the processed data to. The function will save the file as a GeoTIFF
        file with the same name as the original file. The directory structure will be
        preserved and a folder will be create if it does not exist.
    extract_band : str
        Band to extract from the NetCDF file. Following Xu (2022) data, this files have the following possible bands (13 in total):
         - evt_class
         - burn severity(cbi_severity)
         - emissions(CO2, CO, CH4, NMOC, SO2, NH3, NO, NO2, NOx, PM2.5, OC, BC)
    feather : bool, optional
        If True, the function will save a feather file with the long format of the
        emissions data. This is useful for further analysis. Default is False
    overwrite : bool, optional. If True, the function will overwrite existing files.
        This will only work if the NetCDF files are available in the save_path.

    Returns
    -------
    None
        The function will save a GeoTIFF file for each event in the save_path directory
    """

    # List all files in the directory
    events = list(Path(emissions_path).glob("*.grd"))
    # Create save path if not exists
    os.makedirs(save_path, exist_ok=True)

    # Load template details
    template = rioxarray.open_rasterio(template_path)

    # Loop through each tile file and create a new projected resampled file.
    # Append to list if the feather option is on
    emissions_files = []
    for event in tqdm(events, desc="Processing events..."):
        # Create path to temporary file
        path_save_file = os.path.join(save_path, f"{event.stem}.nc")

        # Open file using rioxarray
        event_arr = rioxarray.open_rasterio(event)
        band_dict = {v: k for k, v in enumerate(event_arr.attrs["long_name"])}

        # Subset to the desired band
        event_arr = event_arr.isel(band=band_dict[extract_band])

        # Only proceed if the file does not exist
        if not os.path.exists(path_save_file):
            xr_resampled = (
                xr_reproject(
                    event_arr,
                    geobox=template.geobox,
                    resampling="bilinear",
                    dst_nodata=np.nan,
                )
                .rename({"x": "lon", "y": "lat"})
                .drop_vars(["spatial_ref"])
            )

            xr_resampled.to_netcdf(path_save_file)

            if feather:
                df = xr_resampled.to_dataframe(name=extract_band).dropna().reset_index()

                # Fix name to make it more MTBS-like
                # Warning here: some wildfires are not MTBS, so we need to leave some flexibility
                # when naming the event_id. This has to be fixed by the user! (150 files are weird).
                df["event_id"] = event.stem.replace("_stack", "").upper()
                emissions_files.append(df)
        else:
            print(f"Skipping {event.stem} as it already exists in temporary directory")

    if len(emissions_files) > 0 and feather:
        process_emissions_feather(
            emissions_files, template_path, save_path, extract_band=extract_band
        )
    elif overwrite:
        log.info(f"Reading files in {save_path} to create a feather file")
        emissions_files = list(Path(save_path).glob("*.nc"))
        process_emissions_feather(
            emissions_files, template_path, save_path, extract_band=extract_band
        )
    else:
        log.info(
            "No emission files were processed as feather. Check the input directory"
        )

    return None
