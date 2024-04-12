import os
from pathlib import Path

import geocube
import numpy as np
import pandas as pd
import rasterio
import rioxarray
from odc.algo import xr_reproject
from tqdm import tqdm

from src.utils import prepare_template


def process_dnbr(dnbr_path, template_path, save_path, feather=False):
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

    Returns
    -------
    None
        The function will save a GeoTIFF file for each event in the save_path directory
    """

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
        path_save_file = os.path.join(save_path, f"{event.stem}.nc")

        # Open file using rioxarray
        event = rioxarray.open_rasterio(event)

        # Only proceed if the file does not exist
        if not os.path.exists(path_save_file):
            xr_resampled = (
                xr_reproject(
                    event,
                    geobox=template.geobox,
                    resampling="bilinear",
                    dst_nodata=np.nan,
                )
                .rename({"x": "lon", "y": "lat"})
                .drop_vars(["spatial_ref"])
            )

            xr_resampled.to_netcdf(path_save_file)
        else:
            print(f"Skipping {event.stem} as it already exists in temporary directory")

        if feather:
            df = xr_resampled.to_dataframe(name="dnbr").dropna().reset_index()
            df["event_id"] = event.stem
            dnbr_files.append(df)

    if len(dnbr_files) > 0:
        dnbr_files = pd.concat(dnbr_files)

        # Merge with template
        template = (
            prepare_template(template_path).groupby("grid_id", as_index=False).first()
        )
        dnbr_files = dnbr_files.merge(template, on=["lat", "lon"], how="left")

        dnbr_files.to_feather(os.path.join(save_path, "dnbr_long.feather"))

    return None
