"""Extract and align data to template grid"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from prescribed.extract.process_dem import process_dem
from prescribed.extract.process_disturbances import process_disturbances
from prescribed.extract.process_dnbr import process_dnbr
from prescribed.extract.process_modis_frp import process_modis_file
from prescribed.extract.process_prism import process_variables
from prescribed.extract.process_land_type import process_land_type
from prescribed.extract.process_emissions import process_emissions

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="extract")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    data_extract = cfg.extract.keys()

    # Process the DEM
    if "dem" in data_extract:
        log.info("Processing DEM")
        process_dem(
            dem_path=cfg.extract.dem.path,
            shape_mask=cfg.shape_mask,
            template=cfg.template,
            save_path=cfg.extract.dem.save_path,
            feather=cfg.extract.dem.feather,
        )

    # Process the disturbances
    if "disturbances" in data_extract:
        log.info("Processing disturbances")
        process_disturbances(
            disturbances_path=cfg.extract.disturbances.path,
            template_path=cfg.template,
            save_path=cfg.extract.disturbances.save_path,
            shape_mask=cfg.shape_mask,
            clean=True,
            feather=cfg.extract.disturbances.feather,
            wide=cfg.extract.disturbances.wide,
        )

    # Process the PRISM data
    if "prism" in data_extract:
        log.info("Processing PRISM")
        process_variables(
            variables=cfg.extract.prism.variables,
            path_prism_data=cfg.extract.prism.path,
            save_path=cfg.extract.prism.save_path,
            mask_shape=cfg.shape_mask,
            template=cfg.template,
            feather=cfg.extract.prism.feather,
            wide=cfg.extract.prism.wide,
        )

    # Process the MODIS data
    if "frp" in data_extract:
        log.info("Processing MODIS FRP")
        process_modis_file(
            file_path=cfg.extract.frp.path,
            save_path=cfg.extract.frp.save_path,
            aoi=cfg.shape_mask,
            template_path=cfg.template,
            feather=cfg.extract.frp.feather,
            wide=cfg.extract.frp.wide,
        )

    # Process the DNBR data
    if "dnbr" in data_extract:
        log.info("Processing DNBR")
        process_dnbr(
            dnbr_path=cfg.extract.dnbr.path,
            template_path=cfg.template,
            save_path=cfg.extract.dnbr.save_path,
            feather=cfg.extract.dnbr.feather,
            overwrite=cfg.extract.dnbr.overwrite,
            classes=cfg.extract.dnbr.classes,
        )

    # Process the land type data
    if "land_type" in data_extract:
        log.info("Processing land type")
        process_land_type(
            path=cfg.extract.land_type.path,
            template_path=cfg.template,
            save_path=cfg.extract.land_type.save_path,
        )

    if "emissions" in data_extract:
        log.info("Processing emissions")
        process_emissions(
            emissions_path=cfg.extract.emissions.path,
            template_path=cfg.template,
            save_path=cfg.extract.emissions.save_path,
            extract_band=cfg.extract.emissions.extract_band,
            feather=cfg.extract.emissions.feather,
            overwrite=cfg.extract.emissions.overwrite,
        )


if __name__ == "__main__":
    main()
