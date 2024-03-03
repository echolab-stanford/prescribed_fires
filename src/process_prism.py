import os
import zipfile
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import xarray as xr
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from .transform_array import transform_array_to_xarray


def process_prism(path_to_zip, mask_shape, save_path=None, template=None):
    """Process PRISM compressed data and save as a NetCDF file

    This function takes a zipped yearly file from PRISM and process each month
    file into a NetCDF file. The function also masks the data using a shapefile
    and, if passed, reprojects the files to a common template with a custom
    resolution.

    Parameters
    ----------
    path_to_zip : str
        Path to the zipped PRISM data
    mask_shape : str
        Path to the shapefile to use for masking
    template : str
        Path to the template file to reproject to
    save_path : str
        Path to save the processed data to. The function will create a folder
        with the year and month and save the NetCDF file there.

    Returns
    -------
        A NetCDF file with the processed data
    """

    # Open files is paths are passed instead
    if isinstance(mask_shape, str):
        mask_shape = gpd.read_file(mask_shape)

    # List contents of zip file and take only the bil files
    fs = zipfile.ZipFile(path_to_zip).namelist()
    fs = [f for f in fs if f.endswith(".bil")]

    # Loop over each month file and open using vsizip and rasterio
    monthly_data = []
    for month_file in tqdm(fs, position=1, desc="Processing files..."):
        month_path = f"zip:{path_to_zip}!{month_file}"

        # Get month and year from file name
        date = month_file.split(".")[0].split("_")[-2]

        if len(date) != 6:
            print(f"Skipping {date} as is not a monthly file")
            continue

        # Open zipped raster (bil) using rasterio and ZipMemoryFile
        with rasterio.open(month_path) as src:
            meta = src.meta

            # Project shapefile template
            if mask_shape is not None:
                shapefile_proj = mask_shape.to_crs(meta["crs"])

                # Mask data using shapefile
                data_src, transform_src = rasterio.mask.mask(
                    src, shapes=shapefile_proj.geometry, crop=True
                )

            # Change nasty -9999 values to NaN
            data_src = np.where(data_src == -9999, np.nan, data_src)

            if template:
                # Reproject data to match template bounds and width/height
                # Open template
                with rasterio.open(template) as temp:
                    data = np.zeros(temp.shape, dtype=temp.meta["dtype"])
                    data, transform = reproject(
                        data_src,
                        destination=data,
                        src_transform=transform_src,
                        src_crs=meta["crs"],
                        dst_transform=temp.meta["transform"],
                        dst_crs=temp.meta["crs"],
                        dst_nodata=np.nan,
                        src_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
            else:
                data = data_src

            # Transform to xarray
            data = transform_array_to_xarray(
                data, transform, extra_dims={"time": [datetime.strptime(date, "%Y%m")]}
            )

            # Append to list
            monthly_data.append(data)

        # Append data to yearly xarray
        data_concat = xr.concat(monthly_data, dim="time")

        # Save data
        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            data_concat.to_netcdf(f"save_path/prism_processed_{date}.nc")

    return data_concat
