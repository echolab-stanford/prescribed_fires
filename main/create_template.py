"""Create template grid for aligned data extraction

This script creates a template grid for aligned data extraction. The grid is
defined using the Natural Earth provinces shapefile using user-defined resolution
and projection. The grid is created using the GDAL rasterize command and saved
as a GeoTIFF file. A temporary shapefile is saved in the temp folder.

Example
-------
To create a grid for California with a resolution of 1000 meters using the
albers equal area projection (EPSG:3310), run the following command from the
main directory:

    python create_template.py --path_shapefile /shapefile.shp
    --resolution 1000 --projection 3310 --iso_code US-CA
"""

import os
import geopandas as gpd
import hydra
from omegaconf import DictConfig


def create_grid(
    path_shapefile, resolution, projection, iso_code, save_path="temp"
):
    """Create grid using shapefile and GDAL rasterize command

    Use the Natural Earth shapefile of the world to create a grid for a given
    part of the world. The grid is created using the GDAL rasterize command.
    We use the ISO 3166-2 alpha-3 country codes to select the country of interest

    Parameters
    ----------
    path_shapefile : str
        Path to shapefile
    resolution : float
        Resolution of grid in meters
    projection : int
        EPSG code of projection to use for grid. Default is California meters
        using albers equal area projection (EPSG:3310)
    iso_code : str
        ISO 3166-1 alpha-3 country code for country of interest. Default is
        California USA: US-CA
    save_path : str
        Path to save grid. Default is temp folder in the same directory as the
        script.

    Returns
    -------
    None
    """

    # Create temporary folder for rasterized grid and shape
    os.makedirs(save_path, exist_ok=True)

    # Open shapefile
    gdf = gpd.read_file(path_shapefile)

    # Select country of interest
    gdf = gdf[gdf["iso_3166_2"] == iso_code]

    # Reproject to projection of interest
    gdf = gdf.to_crs(projection)

    # Save to temporary shapefile
    gdf.to_file(os.path.join(save_path, "shapefile_template.shp"))

    # Run gdal_rasterize command to create grid
    os.system(
        f"""
        gdal_rasterize -a_nodata 0 \
            -tr {resolution} {resolution} \
            -burn 1 \
            {os.path.join(save_path, "shapefile_template.shp")} \
            {os.path.join(save_path, "template.tif")}
        """
    )

    return None


@hydra.main(
    config_path="../conf",
    config_name="create_template.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    create_grid(
        path_shapefile=cfg.path_shapefile,
        resolution=cfg.resolution,
        projection=cfg.projection,
        iso_code=cfg.iso_code,
        save_path=cfg.save_path,
    )


if __name__ == "__main__":
    main()
