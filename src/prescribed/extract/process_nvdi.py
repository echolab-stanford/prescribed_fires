import os
from itertools import chain
from pathlib import Path

import shutil
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


def resample_vegetation(files_path, template_path, temporary_path="resampled"):
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


def process_vegetation(
    files_path,
    template_path,
    save_path,
    shape_mask,
    feather,
    wide,
    ncores=-1,
    clean=True,
):
    """Process vegetation data for California across tiles and years

    Function to process vegetation data for California across tiles and years. This function will read in the data, and fix it to a template. Notice that this implies that the data will be transformed and aggregated to a coarser resolution. In all operations we use a `laczos` interpolation and we clip all tile data by year into temporary mosaic files.

    Steps:
    1. Read the data
    2. Mosaic images by year using native resolution
    3. Reproject and resample the data to a template
    4. Clip the data to the shapefile mask
    5. Save the data to a feather file and NetCDF

    We use some multiprocessing to resample big mosaics, the `ncores` option tells the function how many cores to use. If `-1` is passed, the function will use equal number of cores as number of years in the data, this is a good rule of thumb.

    This function will create different intermediate temporary files, if you don't want to keep any of these, you can use the `clean` option.

    Parameters
    ----------
    files_path : str
        Path to the vegetation data. The data should be in a folder as downloaded
    template_path : str
        Path to the template file to reproject to
    save_path : str
        Path to save the processed data to. The function will save the file as a NetCDF
        if a path is passed. The default is not to save
    shape_mask : str
        Path to the shapefile mask to clip the data to
    feather : bool
        If True, save all data as a single feather file in long format, unless wide is True. In that case it will save both file in feather format.
    wide : bool
        If True, save the data in wide format. The file will have the `_wide` suffix. By default the data is saved in long format.
    ncores : int
        Number of cores to use for parallel processing. The default is -1, which will use the number of years in the data.
    clean : bool
        If True, remove all temporary files created during the process. The default is True.

    Returns
    -------
    xr.DataArray
        Fractional vegetation data as an xarray
    """

    # Fractional vegetation categories
    cats = {1: "tree", 2: "shrub", 3: "herbaceous", 4: "bare"}

    # Load template details
    template = rasterio.open(template_path)

    # List all files in the directory
    tiles = [p for p in Path(files_path).glob("*") if p.is_dir()]
    temp_files = [list(p.glob("*.tif")) for p in tiles]

    # Get all files in the same list
    all_files = list(chain(*temp_files))

    dict_years = {k: [] for k in range(1985, 2021)}
    for year, _ in dict_years.items():
        files_year = [f for f in all_files if str(year) in f.stem]
        dict_years[year] = files_year

    if not os.path.exists(os.path.join(save_path, "big_mosaics")):
        os.makedirs(os.path.join(save_path, "big_mosaics"), exist_ok=True)

    for year, files in tqdm(dict_years.items(), desc="Processing years...", position=0):
        # Save file
        save_file = os.path.join(save_path, "big_mosaics", f"year_mosaic_{year}.tif")
        if not os.path.exists(save_file):
            tiles_open = [rasterio.open(tile) for tile in files]

            # Merge tiles and save result as a GeoTIFF
            mosaic, transform = merge(
                tiles_open, method="first", target_aligned_pixels=True
            )

            out_meta = tiles_open[0].meta.copy()
            out_meta.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": transform,
                    "driver": "GTiff",
                }
            )

            with rasterio.open(
                save_file,
                "w",
                **out_meta,
                tiled=True,
                blockxsize=4096,
                blockysize=4096,
            ) as dest:
                dest.write(mosaic)

    # Parallel process the mosaicing of the data
    list_mosaics = list(Path(os.path.join(save_path, "big_mosaics")).glob("*.tif"))
    parallel_resample_vegetation(
        list_mosaics,
        ncores=len(list_mosaics) if ncores == -1 else ncores,
        template_path=template_path,
        save_path=os.path.join(save_path, "resampled"),
    )

    # List all resampled rasters to mask
    resampled_files = list(Path(os.path.join(save_path, "resampled")).glob("*.tif"))

    year_mosaics = []
    for m_year in resampled_files:
        arr = rioxarray.open_rasterio(m_year)
        arr = arr.expand_dims(year=m_year.stem.split("_")[-1])
        year_mosaics.append(arr)

    mosaic = xr.concat(year_mosaics, dim="year")

    # Open mosaic again and mask for the shapefile mask if passed
    # Load shapefile is a string path is passed
    if isinstance(shape_mask, str):
        shape_mask = gpd.read_file(shape_mask)

    clipped = mosaic.rio.clip(
        shape_mask.geometry, shape_mask.crs, drop=True, invert=False
    )

    xr_resampled = xr_reproject(
        clipped,
        geobox=template.geobox,
        resampling="bilinear",
        dst_nodata=arr.attrs["_FillValue"],
    )

    xr_resampled = xr.where(
        xr_resampled >= arr.attrs["_FillValue"], np.nan, xr_resampled
    ).rename({"x": "lon", "y": "lat"})

    # Coverage data is stored as integers. If we use a scale factor of 100, the
    # fraction is from 0 to 100. We want to store coverage as a percentage from
    # 0 to 1, so we use 10000 as a scale factor
    xr_resampled = xr_resampled / 10000

    # Save data if saved is passed
    if feather:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Transform to dataframe
        template_expanded = (
            prepare_template(template_path).groupby("grid_id", as_index=False).first()
        )

        df_resampled = xr_resampled.to_dataframe(name="coverage").reset_index().dropna()

        # Merge with template
        # As other merges with template, remember that is possible to have less rows
        # because of the water pixels and the coastal pixels that are moved in the
        # `xr_reproject`. We want consistency, so we keep them away and always follow
        # the template. See `analysis/weird_points.ipynb` to see the issue.
        df_resampled = df_resampled.merge(
            template_expanded, on=["lat", "lon"], how="right"
        ).drop(columns=["spatial_ref"])

        # Pivot to get bands as column by coverage
        pivot_coverage_columns = pd.pivot(
            df_resampled,
            index=["grid_id", " year", "lat", "lon"],
            columns="band",
            values="coverage",
        )

        pivot_coverage_columns.columns = [
            f"frac_cover_{cats[f]}" for f in pivot_coverage_columns.columns
        ]

        if wide:
            dist_pivot = pd.pivot(
                pivot_coverage_columns.reset_index(),
                index="grid_id",
                columns="year",
                values=[col for col in df_resampled.columns if "frac_cover" in col],
            ).astype(int)

            # Change column names from multiindex to flat
            dist_pivot.columns = [f"{i}_{j}" for i, j in dist_pivot.columns]
            dist_pivot.reset_index(inplace=True)

            # Save data to feather
            dist_pivot.to_feather(
                os.path.join(save_path, "frac_vegetation_wide.feather")
            )

        # Save data
        pivot_coverage_columns.reset_index().to_feather(
            os.path.join(save_path, "frac_vegetation_long.feather")
        )
        xr_resampled.to_netcdf(os.path.join(save_path, "frac_vegetation.nc"))

    if clean:
        shutil.rmtree(os.path.join(files_path, "resampled"))
        shutil.rmtree(os.path.join(files_path, "big_mosaics"))

    return xr_resampled
