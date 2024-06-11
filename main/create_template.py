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
import argparse
import geopandas as gpd


def create_grid(path_shapefile, resolution, projection, iso_code, save_path="temp"):
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


if __name__ == "__main__":
    # Define input arguments
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--path_shapefile",
        required=True,
        type=str,
        help="Path to shapefile",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1000,
        help="Resolution of grid in meters",
    )
    parser.add_argument(
        "--projection",
        type=int,
        default=3310,
        help="EPSG code of projection to use for grid. Default is California meters using albers equal area projection (EPSG:3310)",
    )
    parser.add_argument(
        "--iso_code",
        type=str,
        default="US-CA",
        help="ISO 3166-1 alpha-3 country code for country of interest. Default is California USA: US-CA",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="temp",
        help="Path to save grid. Default is temp folder in the same directory as the script.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Create grid
    create_grid(
        path_shapefile=args.path_shapefile,
        resolution=args.resolution,
        projection=args.projection,
        iso_code=args.iso_code,
        save_path=args.save_path,
    )
