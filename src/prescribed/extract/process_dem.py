import os

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
import xarray as xr
from rasterio.warp import Resampling, reproject
from xrspatial import aspect, curvature, slope

from prescribed.utils import transform_array_to_xarray, prepare_template


def process_dem(dem_path, shape_mask, template, save_path, feather=False):
    """Process DEM to calculate aspect, slope and curvature of the terrain

    This function takes a DEM of any resolution and calculates the aspect, slope
    and curvature of the terrain. The function also masks the data using a
    shapefile and reprojects the files to a common template with a custom
    resolution.

    Parameters
    ----------
    dem_path : str
        Path to the DEM data
    shape_mask : gpd.GeoDataFrame or str
        Path to the shapefile to use for masking or GeoDataFrame object
    template : str
        Path to the template file to reproject to
    save_path : str
        Path to save the processed data to. The function will create a folder
        with the year and month and save the NetCDF file there.
    feather : bool, optional
        If True, save the data as feather files, by default False

    Returns
    -------
    Arrays with landscape charactersitics
    elevation : xarray.DataArray
        Elevation data
    aspect_arr : xarray.DataArray
        Aspect data
    slope_arr : xarray.DataArray
        Slope data
    curvature_arr : xarray.DataArray
        Curvature data
    """

    # Read shapefile if path is provided
    if isinstance(shape_mask, str):
        shape_mask = gpd.read_file(shape_mask)

    # Read DEM
    with rasterio.open(dem_path) as src:
        meta = src.meta

        # Reproject the shape_mask to match the DEM to mask the array
        shapefile_proj = shape_mask.to_crs(meta["crs"])

        # Mask data using shapefile
        data_src, transform_src = rasterio.mask.mask(
            src, shapes=shapefile_proj.geometry, crop=True
        )

        # Remove all nodata to numpy nan
        data_src = np.where(data_src == meta["nodata"], np.nan, data_src)

        # Apply template to transform (align) the data
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
                    dst_nodata=np.nan,
                    src_nodata=np.nan,
                    dst_transform=temp.meta["transform"],
                    dst_crs=temp.meta["crs"],
                    resampling=Resampling.bilinear,
                )
        else:
            data = data_src

        # Transform to xarray
        elevation = transform_array_to_xarray(data, transform).rename("elevation")

        # Calculate aspect, slope and curvature with xarray-spatial
        aspect_arr = aspect(elevation.squeeze()).rename("aspect")
        slope_arr = slope(elevation.squeeze()).rename("slope")
        curvature_arr = curvature(elevation.squeeze()).rename("curvature")

        out = (elevation, aspect_arr, slope_arr, curvature_arr)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if feather:
            # Merge all files since they all have the same affine transform
            merge_arr = xr.merge(out)
            merge_arr = merge_arr.to_dataframe().reset_index().dropna()

            # Get template clean and merge with data
            template = (
                prepare_template(template)
                .groupby("grid_id", as_index=False)
                .first()
                .reset_index()
            )
            merge_arr = merge_arr.merge(template, on=["lat", "lon"]).drop(
                columns=["year"]
            )
            merge_arr.to_feather(os.path.join(save_path, "dem.feather"))

        else:
            # Save data
            aspect_arr.to_netcdf(os.path.join(save_path, "dem_aspect.nc"))
            slope_arr.to_netcdf(os.path.join(save_path, "dem_slope.nc"))
            curvature_arr.to_netcdf(os.path.join(save_path, "dem_curvature.nc"))
            elevation.to_netcdf(os.path.join(save_path, "dem_elevation.nc"))

            out = None

    return out
