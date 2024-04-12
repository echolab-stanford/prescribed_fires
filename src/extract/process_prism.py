import os
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import xarray as xr
from rasterio.warp import Resampling, reproject
from tqdm import tqdm

from src.utils import prepare_template, transform_array_to_xarray


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

            if template is not None:
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


def process_variables(
    variables, path_prism_data, save_path, feather=False, wide=False, **kwargs
):
    """Process all PRISM variables and save as NetCDF files

    Notice that if wide is `True`, the function will save the data in both wide and long formats.

    Parameters
    ----------
    variables : list
        List with the variables to process
    path_prism_data : str
        Path to the PRISM data
    save_path : str
        Path to save the processed data to. The function will create a folder
        with the year and month and save the NetCDF file there.
    template : str
        Path to template data. This is used in this code to add a `grid_id` variable to the output feather files. Is not used for resampling.
    feather: bool
        If True, the function will save the data as feather files instead of NetCDF
    wide: bool
        If True, the function will save the data as a wide format. If wide is True, then both feather and NetCDF files will be saved.
    **kwargs : dict
        Extra arguments to pass to the process_prism function

    Returns
    -------
        None. Saves NetCDF files with the processed data
    """

    data_prism = list(Path(path_prism_data).rglob("*.zip"))
    filter_data_prism = [f for f in data_prism if "all" in f.name]

    for var in tqdm(variables, position=0, desc="Processing PRISM variables..."):
        filter_var_prism = [f for f in filter_data_prism if var in f.name]

        yearly_var_data = []
        for f in filter_var_prism:
            data = process_prism(
                path_to_zip=f,
                **kwargs,
            )
            yearly_var_data.append(data)

        # Concat data
        data_concat = (
            xr.concat(yearly_var_data, dim="time").sortby("time").to_dataset(name=var)
        )

        # Save data
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        data_concat.to_netcdf(os.path.join(save_path, f"prism_processed_{var}.nc"))

    if feather:
        print("Processing PRISM as a long feather file...")

        # Load template line
        template = (
            prepare_template(kwargs["template"])
            .groupby("grid_id", as_index=False)
            .first()
        )

        prism_data = [
            xr.open_dataset(
                os.path.join(save_path, f"prism_processed_{var}.nc"),
                chunks={"time": 10},
            )
            for var in variables
        ]
        prism_all = xr.merge(prism_data)

        df_prism = prism_all.to_dataframe().reset_index().dropna()
        df_prism = df_prism.assign(
            year=df_prism["time"].dt.year, month=df_prism["time"].dt.month
        )

        df_prism = df_prism.merge(
            template[["grid_id", "lat", "lon"]], on=["lat", "lon"], how="inner"
        )
        df_prism.to_feather(os.path.join(save_path, "prism_processed_long.feather"))

        if wide:
            print("Processing PRISM as a wide feather file...")
            df_prism.drop(columns=["year", "month"], inplace=True)
            df_prism = pd.pivot(
                df_prism,
                index=["lat", "lon", "grid_id"],
                columns="time",
                values=["tmin", "tmax", "tdmean", "vpdmin", "vpdmax", "ppt", "tmean"],
            )
            df_prism.columns = [
                f"{i}_{j.strftime('%Y_%m')}" for i, j in df_prism.columns
            ]
            df_prism.reset_index(inplace=True)
            df_prism.to_feather(os.path.join(save_path, "prism_processed_wide.feather"))

    return None
