import os
import logging
import hydra
from omegaconf import DictConfig
from prescribed.build.build_data import (
    fill_treatment_template_mtbs,
    fill_treatment_template_frp,
    build_lhs,
    treatment_schedule,
)

log = logging.getLogger(__name__)


@hydra.main(
    config_path="../conf",
    version_base=None,
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    log.info(f"Building dataset using {cfg.build.treat_type}")

    if cfg.build.treat_type == "mtbs":
        rhs = fill_treatment_template_mtbs(
            template_path=cfg.template,
            treatments_path=cfg.build.treatments_path,
            query=cfg.build.query,
            min_count_treatments=cfg.build.min_count_treatments,
            verbose=cfg.build.verbose,
        )
    elif cfg.build.treat_type == "frp":
        rhs = fill_treatment_template_frp(
            template_path=cfg.template,
            treatments_path=cfg.build.treatments_path,
            frp=cfg.build.frp,
        )
    else:
        raise ValueError(f"Unknown treatment type: {cfg.build.treat_type}")

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

    treatments.to_feather(
        os.path.join(
            cfg.build.save_path, f"wide_treats_{cfg.build.treat_type}.feather"
        )
    )


if __name__ == "__main__":
    main()
