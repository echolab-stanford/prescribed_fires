""" Extract and align data to template grid

Extract all data to a processed directory with all the data aligned to a template
and masked using a concondant shapefile. This scrip will just execute the functions
in the source directory to process:
    - DEM (Elevation, Slope, Aspect, Curvature)
    - Disturbances
    - PRISM 
    - MODIS FRP

Data will be saved in the following directory structure:
    - processed
        - dem
            - elevation.nc
            - slope.nc
            - aspect.nc
            - curvature.nc
        - disturbances
            - disturbances.nc
        - prism
            - tmin.nc
            - tmax.nc
            - tdmean.nc
            - vpdmin.nc
            - vpdmax.nc
            - ppt.nc
            - tmean.nc
        - modis
            - frp (files per year)
        - dnbr
            - dnbr (files per event) in GeoTIFF format
"""

import argparse
import os
from pathlib import Path

import xarray as xr
from tqdm import tqdm

from src.process_dem import process_dem
from src.process_disturbances import process_disturbances
from src.process_prism import process_prism
from src.process_modis_frp import process_modis_file
from src.process_dnbr import process_dnbr


def process_variables(variables, path_prism_data, save_path, **kwargs):
    """Process all PRISM variables and save as NetCDF files

    Parameters
    ----------
    variables : list
        List with the variables to process
    path_prism_data : str
        Path to the PRISM data
    save_path : str
        Path to save the processed data to. The function will create a folder
        with the year and month and save the NetCDF file there.
    **kwargs : dict
        Extra arguments to pass to the process_prism function

    Returns
    -------
        None. Saves NetCDF files with the processed data
    """

    data_prism = list(Path(path_prism_data).rglob("*.zip"))
    filter_data_prism = [f for f in data_prism if "all" in f.name]

    for var in tqdm(variables, position=0, desc="Processing variables..."):
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


if __name__ == "__main__":
    # Define the parser
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to the template file to reproject to",
        required=True,
    )
    parser.add_argument(
        "--shapefile",
        type=str,
        help="Path to the shapefile to use for masking",
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the processed data to",
        required=True,
    )
    parser.add_argument(
        "--path_to_dem", type=str, help="Path to the DEM data to process", default=None
    )
    parser.add_argument(
        "--path_to_disturbances",
        type=str,
        help="Path to the disturbances data to process",
        default=None,
    )
    parser.add_argument(
        "--path_to_prism",
        type=str,
        help="Path to the PRISM data to process",
        default=None,
    )
    parser.add_argument(
        "--path_to_modis",
        type=str,
        help="Path to the MODIS data to process",
        default=None,
    )
    parser.add_argument(
        "--path_to_dnbr",
        type=str,
        help="Path to the DNBR data to process",
        default=None,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Process the DEM
    if args.path_to_dem is not None:
        process_dem(
            dem_path=args.path_to_dem,
            shape_mask=args.shapefile,
            template=args.template,
            save_path=os.path.join(args.save_path, "dem"),
        )

    # Process the disturbances
    if args.path_to_disturbances is not None:
        process_disturbances(
            disturbances_path=args.path_to_disturbances,
            template_path=args.template,
            save_path=os.path.join(args.save_path, "disturbances"),
            shape_mask=args.shapefile,
            clean=True,
        )

    # Process the PRISM data
    if args.path_to_prism is not None:
        process_variables(
            variables=["tmin", "tmax", "tdmean", "vpdmin", "vpdmax", "ppt", "tmean"],
            path_prism_data=args.path_to_prism,
            save_path=os.path.join(args.save_path, "prism"),
            mask_shape=args.shapefile,
            template=args.template,
        )

    # Process the MODIS data
    if args.path_to_modis is not None:
        process_modis_file(
            file_path=args.path_to_modis,
            save_path=os.path.join(args.save_path, "frp"),
            aoi=args.shapefile,
            template=args.template,
        )

    # Process the DNBR data
    if args.path_to_dnbr is not None:
        process_dnbr(
            dnbr_path=args.path_to_dnbr,
            template_path=args.template,
            save_path=os.path.join(args.save_path, "dnbr_template"),
        )
