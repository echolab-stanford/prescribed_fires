import os

import numpy as np
import rioxarray
import xarray as xr
from odc.algo import xr_reproject

from ..utils import prepare_template


def process_land_type(path: str, template_path: str, save_path: str):
    """Process land type data to common template"""

    # Load template
    template = rioxarray.open_rasterio(template_path)

    land_type = rioxarray.open_rasterio(
        path,
        chunks={"x": 1000, "y": 1000},
    )

    xr_resampled = xr_reproject(
        land_type, geobox=template.geobox, resampling="mode", dst_nodata=255
    )
    xr_resampled = xr.where(xr_resampled >= 254, np.nan, xr_resampled).rename(
        {"x": "lon", "y": "lat"}
    )

    # Transform to dataframe
    template_expanded = (
        prepare_template(template_path).groupby("grid_id", as_index=False).first()
    )

    df_resampled = xr_resampled.to_dataframe(name="land_type").reset_index().dropna()

    df_resampled = df_resampled.merge(template_expanded, on=["lat", "lon"]).drop(
        columns=["year", "band", "spatial_ref"]
    )

    # Save to feather
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_resampled.to_feather(os.path.join(save_path, "land_type.feather"))
