import logging
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import odc.geo.xr
import pandas as pd
import rioxarray
import xarray as xr
from odc.algo import xr_reproject
from tqdm import tqdm

from prescribed.utils import prepare_template

log = logging.getLogger(__name__)


def process_dnbr_feather(
    list_files: List[Union[Path, pd.DataFrame]], template_path: str, save_path: str
) -> None:
    """Aux file to process list of NetCDFs to feather file

    This function takes a file of dnbr files in NetCDF format and reads them into
    feather format while adding the event_id in the process. The function will either
    take a list of files or a list of open objects in feather format.

    Parameters
    ----------
    list_files : list
        List of files to process. The files can be either a string with paths or a set of pd.Dataframe objects.
    template_path : str
        Path to the template file to reproject to. The template should be a GeoTIFF
    save_path : str
        Path to save the feather file to
    """

    # Check all elements in the list are a dataframe
    if all([isinstance(f, pd.DataFrame) for f in list_files[1:5]]):
        df = pd.concat(list_files)
    else:
        list_objs = []
        for p in tqdm(list_files, desc="Reading nc4 files into feather..."):
            df = xr.open_dataarray(p).to_dataframe("dnbr").dropna().reset_index()
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
    dnbr_files = df.merge(template, on=["lat", "lon"])

    # If grid_id and year are floats, transform to int
    dnbr_files["grid_id"] = dnbr_files["grid_id"].astype(int)

    if "band" in dnbr_files.columns:
        dnbr_files.drop(columns=["band"], inplace=True)

    log.info(f"Saving file into {os.path.join(save_path, 'dnbr_long.feather')}")
    dnbr_files.to_feather(os.path.join(save_path, "dnbr_long.feather"))

    return None


def process_dnbr(
    dnbr_path, template_path, save_path, feather=False, overwrite=False, classes=False
):
    """Process DNBR data for template

    Process DNBR data (@ 30m resolution) to overlay with a user-defined template.
    The function expects separate GeoTIFF files for each event. The function will
    read each file and resample it to fit the template grid and projection. Each
    file will be saved with the same name as the original file in the save_path
    directory and the directory structure will be preserved.

    Parameters
    ----------
    dnbr_path : str
        Path to the DNBR data. The data format should be GeoTIFF
    template_path : str
        Path to the template file to reproject to. The template should be a GeoTIFF
        file with the desired projection and grid
    save_path : str
        Path to save the processed data to. The function will save the file as a GeoTIFF
        file with the same name as the original file. The directory structure will be
        preserved and a folder will be create if it does not exist.
    feather : bool, optional
        If True, the function will save a feather file with the long format of the
        DNBR data. This is useful for further analysis. Default is False
    overwrite : bool, optional. If True, the function will overwrite existing files.
        This will only work if the NetCDF files are available in the save_path.
    class: bool, optional
        Is this a class raster? If it is, then the aggregation into the template
        should be using a "mode" or "nearest", rather than a "bilinear" interpolation.

    Returns
    -------
    None
        The function will save a GeoTIFF file for each event in the save_path directory
    """

    # Select interpolation for sampling
    if classes:
        resampling_mode = "mode"
    else:
        resampling_mode = "bilinear"

    # List all files in the directory
    events = list(Path(dnbr_path).glob("*.tif"))
    # Create save path if not exists
    os.makedirs(save_path, exist_ok=True)

    # Load template details
    template = rioxarray.open_rasterio(template_path)

    # Loop through each tile file and create a new projected resampled file.
    # Append to list if the feather option is on
    dnbr_files = []
    for event in tqdm(events, desc="Processing events..."):
        # Create path to temporary file

        # Clean stem in case some data comes with additional stuff. We assume in
        # here that each file is using a 21-long ID like the MTBS ones. Thus, then
        # we can use the stem to merge with the template
        stem = event.stem.upper()

        if len(stem) == 21:
            path_save_file = os.path.join(save_path, f"{stem}.nc")

            # Open file using rioxarray
            event_arr = rioxarray.open_rasterio(event)

            # Only proceed if the file does not exist
            if not os.path.exists(path_save_file):
                xr_resampled = (
                    xr_reproject(
                        event_arr,
                        geobox=template.geobox,
                        resampling=resampling_mode,
                        dst_nodata=np.nan,
                    )
                    .rename({"x": "lon", "y": "lat"})
                    .drop_vars(["spatial_ref"])
                )

                xr_resampled.to_netcdf(path_save_file)

                if feather:
                    df = xr_resampled.to_dataframe(name="dnbr").dropna().reset_index()
                    df["event_id"] = stem
                    dnbr_files.append(df)
            else:
                log.info(f"Skipping {stem} as it already exists in temporary directory")
        else:
            log.info(f"Skipping not correct named files: {stem}")

    if len(dnbr_files) > 0 and feather:
        process_dnbr_feather(dnbr_files, template_path, save_path)
    elif overwrite:
        log.info(f"Reading files in {save_path} to create a feather file")
        dnbr_files = list(Path(save_path).glob("*.nc"))
        process_dnbr_feather(dnbr_files, template_path, save_path)
    else:
        log.info("No DNBR files were processed as feather. Check the input directory")

    return None
