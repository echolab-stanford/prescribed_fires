import logging
import hydra
from omegaconf import DictConfig
from prescribed.build.create_distances import create_distances

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf",
    version_base=None,
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    log.info(
        "Building dataset using distances. This is meant only for spillovers"
    )

    for dist in cfg.build.pop_buffer:
        log.info(f"Building dataset using {dist}")
        create_distances(
            mtbs_shapefile=cfg.data.mtbs,
            template=cfg.data.template,
            buffer=dist,
            pop_threshold=cfg.build.pop_threshold,
            buffer_treatment=dist,
            pop_raster_path=cfg.build.pop_raster_path,
            mask=cfg.data.mask,
            save_path=cfg.compute.save_path,
        )


if __name__ == "__main__":
    main()
