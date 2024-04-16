import os
import logging
import hydra
from omegaconf import DictConfig
from prescribed.build.build_data import (
    fill_treatment_template,
    build_lhs,
    treatment_schedule,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="build")
def main(cfg: DictConfig) -> None:
    log.info("Building dataset")
    rhs = fill_treatment_template(
        template_path=cfg.template,
        treatments_path=cfg.build.treatments_path,
        query=cfg.build.query,
        staggered=cfg.build.staggered,
        min_count_treatments=cfg.build.min_count_treatments,
        verbose=cfg.build.verbose,
    )

    log.info("Building LHS: FRP and dNBR")
    lhs = build_lhs(
        covariates_dict=cfg.build.lhs,
    )

    # Merge both datasets
    treatments = treatment_schedule(rhs, lhs)

    # Save the dataset
    log.info(f"Saving dataset to {cfg.build.save_path}")
    if not os.path.exists(cfg.build.save_path):
        os.mkdir(cfg.build.save_path)

    treatments.to_feather(os.path.join(cfg.build.save_path, "wide_treats.feather"))


if __name__ == "__main__":
    main()
