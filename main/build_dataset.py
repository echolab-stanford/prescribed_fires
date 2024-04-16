import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from prescribed.build.build_data import build_dataset, build_lhs, treatment_schedule

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="build")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    log.info("Building dataset")
    rhs = build_dataset(
        template_path=cfg.template,
        treatments_path=cfg.build.dataset.treatments_path,
        query=cfg.build.dataset.query,
        staggered=cfg.build.dataset.staggered,
        min_count_treatments=cfg.build.dataset.min_count_treatments,
        verbose=cfg.build.dataset.verbose,
    )

    log.info("Building LHS")
    lhs = build_lhs(
        template_path=cfg.template,
        covariates_dict=cfg.build.lhs.covariates_dict,
    )

    # Merge both datasets
    treatments = treatment_schedule(rhs, lhs)

    # Save the dataset
    log.info(f"Saving dataset to {cfg.build.save_path}")
    treatments.to_feather(cfg.build.save_path)


if __name__ == "__main__":
    main()