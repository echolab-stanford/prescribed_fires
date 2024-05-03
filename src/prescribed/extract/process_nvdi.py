import os
import shutil
from itertools import chain
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from joblib import Parallel, delayed
from odc.algo import xr_reproject
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from tqdm import tqdm

from prescribed.utils import prepare_template


def parallel_resample_vegetation(path_to_dists, ncores, **kwargs):
    """Parallel wrapper for resample_vegetation function.

    Function to parallelize the resample_vegetation function using joblib.

    Parameters
    ----------
    path_to_dists : list
        List of paths to the files to process
    ncores : int
        Number of cores to use for parallel processing
    **kwargs : dict
        Additional arguments to pass to the resample_vegetation function

    Returns
    -------
    None
    """
    Parallel(n_jobs=ncores)(
        delayed(resample_vegetation)(path, **kwargs) for path in path_to_dists
    )

    return None


def parallel_mosaic_processing(tiles, ncores, **kwargs):
    """Parallel wrapper for the mosaic_processing function.

    Function to parallelize the mosaic_processing function using joblib.

    Parameters
    ----------
    tiles : dict
        Dict with paths as a list to the files to process with a key:
        {2004: [<path>, ..., <path>], 2005: [<path>, ..., <path>], ...}
    ncores : int
        Number of cores to use for parallel processing
    **kwargs : dict
        Additional arguments to pass to the mosaic_processing function

    Returns
    -------
        None
    """

    Parallel(n_jobs=ncores)(
        delayed(mosaic_processing)(tile, year, **kwargs) for year, tile in tiles.items()
    )

    return None


def resample_vegetation(files_path, template_path, temporary_path="temp"):
    """Process fractional vegatation data for California.

    Process fractional vegetation data (@ 30m resolution) for California. The
    data comes in separated files of 5000 x 5000 pixels tiles. To process each
    tile, we are first reprojecting the data to a common template and then storing
    these projection as an intermediate process. Once all tiles are reprojected,
    we merge them into a single file and save as a NetCDF file.

    This process will avoid having to reproject the data at its native resolution
    to the template projection and grid, which is a very slow process.

    The structure of the data is as follows:
        - Directories contain data by tile and each file is a year of data. The data is
        in GeoTIFF format with each band representing a different vegetation type:
        1:tree, 2: shrub 3: herbaceous and 4: bare

    Data source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/KMBYYM#

    Parameters
    ----------
    files_path : str
        Path to the vegetation data. The data should be in a folder as downloaded
    template_path : str
        Path to the template file to reproject to
    save_path : str
        Path to save the processed data to. The function will save the file as a NetCDF
        if a path is passed. The default is not to save
    temporary_path : str
        Path to save the temporary files to. The default is "temp"

    Returns
    -------
    None
        Saves the processed data to the save_path if a path is passed
    """

    # Create temporary directory
    os.makedirs(os.path.join(files_path, temporary_path), exist_ok=True)

    # List all files in the directory
    tiles = list(Path(files_path).glob("*.tif"))

    # Load template details
    template = rasterio.open(template_path)
    template_meta = template.meta
    resolution = template_meta["transform"].a

    # Loop through each year file for individual tile file and create a new
    # projected and resampled file in /temp
    for tile_year in tqdm(tiles, desc="Processing tiles...", position=0):
        # Create path to temporary file
        path_temp_file = os.path.join(
            files_path, temporary_path, f"{tile_year.stem}.tif"
        )

        # Only proceed if the file does not exist
        if not os.path.exists(path_temp_file):
            with rasterio.open(tile_year) as src:
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
                            resampling=Resampling.lanczos,
                        )
        else:
            print(
                f"Skipping {tile_year.stem} as it already exists in temporary directory"
            )

    return None


def mosaic_processing(
    tile_list, key, template_bounds, template_meta, temporary_path="mosaics"
):
    """Process list of GeoTIFF files and merge them into a single file as as mosaic

    This function process a list of closed or opened GeoTiffs using rasterio and
    merge them into a single mosaic file per year without any masking. This function
    meants to be run in parallel with the `parallel_mosaic_processing`

    Parameters
    ----------
    tile_list : list
        List of rasterio opened files to merge
    key : str
        Key to use to save the file. We use a dict by year, so ideally is a year string.
    template_bounds : tuple
        Bounds of the template file to merge the data to
    template_meta : dict
        Template metadata. This is `rasterio.open(template_path).meta`
    temporary_path : str
        Path to save the temporary files to. The default is "mosaics"

    Returns
    -------
    None
        Saves file to temporary directory
    """

    # Fractional vegetation categories
    # 1:tree, 2: shrub 3: herbaceous and 4: bare
    # cats = {1: "tree", 2: "shrub", 3: "herbaceous", 4: "bare"}

    # Check saving folder
    if not os.path.exists(temporary_path):
        os.makedirs(temporary_path, exist_ok=True)

    tiles_open = [rasterio.open(tile) for tile in tile_list]

    # Merge tiles
    mosaic, transform = merge(
        tiles_open,
        bounds=template_bounds,
        resampling=Resampling.lanczos,
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
        os.path.join(temporary_path, f"mosaic_{key}.tif"), "w", **out_meta
    ) as dst:
        dst.write(mosaic)

    return None


def process_vegetation(
    files_path,
    template_path,
    ncores,
    save_path,
    shape_mask,
    wide,
    clean,
    **kwargs,
):
    """Process vegetation data for California across tiles and years

    Function to process vegetation data for California across tiles and years.

    Parameters
    ----------
    files_path : str
        Path to the vegetation data. The data should be in a folder as downloaded
    ncores : int
        Number of cores to use for parallel processing
    **kwargs : dict
        Additional arguments to pass to the resample_vegetation function

    Returns
    -------
    None
    """

    # Load template details
    template = rasterio.open(template_path)
    template_meta = template.meta
    template_bounds = template.bounds

    # List all files in the directory
    tiles = [p for p in Path(files_path).glob("*") if p.is_dir()]

    # Parallel process the resampling of the data
    parallel_resample_vegetation(tiles, ncores, **kwargs)

    temp_dirs = list(Path(files_path).rglob("temp"))
    temp_files = [list(p.glob("*.tif")) for p in temp_dirs]

    # Get all files in the same list
    all_files = list(chain(*temp_files))

    dict_years = {k: [] for k in range(1985, 2021)}
    for year, _ in dict_years.items():
        files_year = [f for f in all_files if str(year) in f.stem]
        dict_years[year] = files_year

    # Parallel process the mosaicing of the data
    parallel_mosaic_processing(dict_years, ncores, template_bounds, template_meta)

    # Open mosaic again and mask for the shapefile mask if passed
    # Load shapefile is a string path is passed
    if isinstance(shape_mask, str):
        shape_mask = gpd.read_file(shape_mask)

    # Load all mosaics as arrays
    mosaic_files = list(Path(os.path.join(save_path, "mosaics")).glob("*.tif"))

    year_mosaics = []
    for m_year in mosaic_files:
        arr = rioxarray.open_rasterio(m_year)
        arr = arr.expand_dims(year=[m_year.stem.split("_")[-1].split(".")[0]])
        year_mosaics.append(arr)

    mosaic = xr.concat(year_mosaics, dim="year")

    xr_resampled = xr_reproject(
        mosaic,
        geobox=template.geobox,
        resampling="bilinear",
        dst_nodata=arr.attrs["_FillValue"],
    )

    xr_resampled = xr.where(xr_resampled >= 254, np.nan, xr_resampled).rename(
        {"x": "lon", "y": "lat"}
    )

    # Save data if saved is passed
    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Transform to dataframe
        template_expanded = (
            prepare_template(template_path).groupby("grid_id", as_index=False).first()
        )

        df_resampled = xr_resampled.to_dataframe(name="coverage").reset_index().dropna()

        # Merge with template
        df_resampled = df_resampled.merge(template_expanded, on=["lat", "lon"]).drop(
            columns=["year", "band", "spatial_ref"]
        )

        # Pivot to get bands as column by coverage
        pivot_coverage_columns = pd.pivot(
            df_resampled,
            index="grid_id",
            columns="band",
            values="coverage",
        )

        pivot_coverage_columns.columns = [
            f"{i}_{j.strftime('%Y_%m')}" for i, j in pivot_coverage_columns.columns
        ]

        if wide:
            dist_pivot = pd.pivot(
                df_resampled,
                index="grid_id",
                columns="year",
                values=[col for col in df_resampled.columns if "disturbances" in col],
            ).astype(int)

            # Change column names from multiindex to flat
            dist_pivot.columns = [f"{i}_{j}" for i, j in dist_pivot.columns]
            dist_pivot.reset_index(inplace=True)

            # Save data to feather
            dist_pivot.to_feather(os.path.join(save_path, "disturbances_wide.feather"))

        # Save data
        arr.to_netcdf(os.path.join(save_path, "disturbances.nc"))

    return arr
