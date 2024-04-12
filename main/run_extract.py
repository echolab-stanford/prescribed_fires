"""Extract and align data to template grid

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

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src.extract.process_dem import process_dem
from src.extract.process_disturbances import process_disturbances
from src.extract.process_dnbr import process_dnbr
from src.extract.process_modis_frp import process_modis_file
from src.extract.process_prism import process_variables


@hydra.main(config_path="conf", config_name="extract")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Process the DEM
    if cfg.path_to_dem is not None:
        process_dem(
            dem_path=cfg.path_to_dem,
            shape_mask=cfg.shapefile,
            template=cfg.template,
            save_path=os.path.join(cfg.save_path, "dem"),
            feather=cfg.feather,
        )

    # Process the disturbances
    if cfg.path_to_disturbances is not None:
        process_disturbances(
            disturbances_path=cfg.path_to_disturbances,
            template_path=cfg.template,
            save_path=os.path.join(cfg.save_path, "disturbances"),
            shape_mask=cfg.shapefile,
            clean=True,
            feather=cfg.feather,
            wide=cfg.wide,
        )

    # Process the PRISM data
    if cfg.path_to_prism is not None:
        process_variables(
            variables=["tmin", "tmax", "tdmean", "vpdmin", "vpdmax", "ppt", "tmean"],
            path_prism_data=cfg.path_to_prism,
            save_path=os.path.join(cfg.save_path, "prism"),
            mask_shape=cfg.shapefile,
            template=cfg.template,
            feather=cfg.feather,
            wide=cfg.wide,
        )

    # Process the MODIS data
    if cfg.path_to_modis is not None:
        process_modis_file(
            file_path=cfg.path_to_modis,
            save_path=os.path.join(cfg.save_path, "frp"),
            aoi=cfg.shapefile,
            template=cfg.template,
            feather=cfg.feather,
            wide=cfg.wide,
        )

    # Process the DNBR data
    if cfg.path_to_dnbr is not None:
        process_dnbr(
            dnbr_path=cfg.path_to_dnbr,
            template_path=cfg.template,
            save_path=os.path.join(cfg.save_path, "dnbr_template"),
            feather=cfg.feather,
        )


if __name__ == "__main__":
    main()
