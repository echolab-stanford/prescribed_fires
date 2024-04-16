import os
import shutil
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm

from prescribed.utils import transform_array_to_xarray, prepare_template


def process_disturbances(
    disturbances_path,
    template_path,
    save_path=None,
    temporary_path="temp",
    shape_mask=None,
    clean=False,
    feather=False,
    wide=False,
):
    """Process disturbances data for California.

    Process disturbances vegetation data (@ 30m resolution) for California. The
    data comes in separated files of 5000 x 5000 pixels tiles. To process each
    tile, we are first reprojecting the data to a common template and then storing
    these projection as an intermediate process. Once all tiles are reprojected,
    we merge them into a single file and save as a NetCDF file.

    This process will avoid having to reproject the data at its native resolution
    to the template projection and grid, which is a very slow process.

    Data source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/CVTNLY

    Parameters
    ----------
    disturbances_path : str
        Path to the disturbances data. The data should be in a folder as downloaded
    template_path : str
        Path to the template file to reproject to
    save_path : str
        Path to save the processed data to. The function will save the file as a NetCDF
        if a path is passed. The default is not to save
    temporary_path : str
        Path to save the temporary files to. The default is "temp"
    shape_mask : str
        Path to the shapefile to use for masking. The default is no masking. So data will
        be collapsed as it comes. Notice some of the tiles have data for states other than
        Califonia, so masking is recommended.
    clean : bool
        If True, the function will remove the temporary directory and the mosaic file.
    feather: bool
        If True, save the data a unique feather, by default False
    wide: bool
        If True, save the data in a wide format, by default False. Notice that this can only work if feather is `True` as well.

    Returns
    -------
    xarray.DataArray or None
        An xarray with the processed data
    """

    # Create temporary directory
    os.makedirs(os.path.join(disturbances_path, temporary_path), exist_ok=True)

    # List all files in the directory
    tiles = list(Path(disturbances_path).glob("*.tif"))

    # Load template details
    template = rasterio.open(template_path)
    template_meta = template.meta
    template_bounds = template.bounds
    resolution = template_meta["transform"].a

    # Loop through each tile file and create a new projected resampled file in /temp
    for tile in tqdm(tiles, desc="Processing tiles...", position=0):
        # Create path to temporary file
        path_temp_file = os.path.join(
            disturbances_path, temporary_path, f"{tile.stem}.tif"
        )

        # Only proceed if the file does not exist
        if not os.path.exists(path_temp_file):
            with rasterio.open(tile) as src:
                # Load meta
                meta = src.meta

                # Calculate bounds and transform for template
                transform, width, height = calculate_default_transform(
                    src.crs,
                    template_meta["crs"],
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=resolution,
                )

                # Update meta with new bounds and transform for the old source
                kwargs = src.meta.copy()
                kwargs.update(
                    {
                        "crs": template_meta["crs"],
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                # Resample data to template projection and grid and save to temporary file
                with rasterio.open(path_temp_file, "w", **kwargs) as dst:
                    for i in tqdm(
                        range(1, src.count + 1),
                        position=1,
                        desc="Resampling bands...",
                    ):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=meta["crs"],
                            src_nodata=meta["nodata"],
                            dst_nodata=meta["nodata"],
                            num_threads=5,  # TODO: Make this an option
                            dst_crs=template_meta["crs"],
                            resampling=Resampling.nearest,
                        )
        else:
            print(f"Skipping {tile.stem} as it already exists in temporary directory")

            # Load meta from tile to fill into the mosaic
            with rasterio.open(tile) as src:
                meta = src.meta

    # List all files in the temporary directory
    tiles = list(Path(os.path.join(disturbances_path, temporary_path)).glob("*.tif"))
    tiles_open = [rasterio.open(tile) for tile in tiles]

    # Merge tiles
    mosaic, transform = merge(
        tiles_open,
        bounds=template_bounds,
        resampling=Resampling.nearest,
    )

    # Save mosaic in the temporary directory as a GeoTIFF. Update dict to keep
    # the important metadata from the original files.

    # Update metadata
    out_meta = tiles_open[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": transform,
            "crs": template_meta["crs"],
        }
    )

    with rasterio.open(
        os.path.join(disturbances_path, temporary_path, "mosaic.tif"), "w", **out_meta
    ) as dst:
        dst.write(mosaic)

    # Close files to avoid memory issues
    [tile.close() for tile in tiles_open]

    # Open mosaic again and mask for the shapefile mask if passed
    if shape_mask is not None:
        # Load shapefile is a string path is passed
        if isinstance(shape_mask, str):
            shape_mask = gpd.read_file(shape_mask)

        with rasterio.open(
            os.path.join(disturbances_path, temporary_path, "mosaic.tif")
        ) as src:
            # Project shapefile template
            shapefile_proj = shape_mask.to_crs(src.meta["crs"])

            # Mask data using shapefile
            mosaic, transform = rasterio.mask.mask(
                src,
                shapes=shapefile_proj.geometry,
                crop=True,
                nodata=src.meta["nodata"],
            )

    # Create array of dates: data starts in 1985 to 2021. Each file band is
    # an individual year
    if mosaic.shape[0] != 37:
        raise ValueError(
            f"Expected number of years (37) does not match number of raster bands ({mosaic.shape[0]})"
        )

    year_arr = np.arange(1985, 2022)

    # Remove all nodata to numpy nan
    mosaic = np.where(mosaic == meta["nodata"], np.nan, mosaic)

    # Transpose array to match the expected dimensions
    mosaic = np.transpose(mosaic, axes=(1, 2, 0))

    # Save mosaic as a NetCDF file with all dates
    arr = transform_array_to_xarray(mosaic, transform, extra_dims={"time": year_arr})

    # Save data if saved is passed
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if feather:
            # Template to expand the data
            template_expanded = prepare_template(template_path)

            disturbances = arr.squeeze()

            dist_df = (
                disturbances.to_dataframe(name="disturbances").reset_index().dropna()
            )

            # Merge with template
            dist_df.rename(columns={"time": "year"}, inplace=True)
            dist_df = dist_df.merge(
                template_expanded, on=["lat", "lon", "year"], how="right"
            )
            dist_df.fillna(0, inplace=True)

            # Replace disturbance values for the actual disturbance names
            dist_df.disturbances = dist_df.disturbances.replace(
                {
                    0: "no_disturbance",
                    1: "fire",
                    2: "timber_harvest",
                    3: "drought_forest_die",
                    4: "unattributed_greening",
                    5: "unattributed_browning",
                }
            )

            # Create dummies in the long format and then make it wider
            dist_df_dummies = pd.get_dummies(dist_df, columns=["disturbances"])
            dist_df_dummies.to_feather(
                os.path.join(save_path, "disturbances_long.feather")
            )

            if wide:
                dist_pivot = pd.pivot(
                    dist_df_dummies,
                    index="grid_id",
                    columns="year",
                    values=[
                        col for col in dist_df_dummies.columns if "disturbances" in col
                    ],
                ).astype(int)

                # Change column names from multiindex to flat
                dist_pivot.columns = [f"{i}_{j}" for i, j in dist_pivot.columns]
                dist_pivot.reset_index(inplace=True)

                # Save data to feather
                dist_pivot.to_feather(
                    os.path.join(save_path, "disturbances_wide.feather")
                )

        # Save data
        arr.to_netcdf(os.path.join(save_path, "disturbances.nc"))

    # Clean the house (remove temporary files and mosaic)
    if clean:
        shutil.rmtree(os.path.join(disturbances_path, temporary_path))

    return arr
