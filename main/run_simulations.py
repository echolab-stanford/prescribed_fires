"""Run simulations for different parameters"""

import logging

import geopandas as gpd
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from prescribed.estimate.simulations import (
    run_simulations,
    simulation_data,
)
from prescribed.utils import prepare_template

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    sim_data = simulation_data(
        template=cfg.data.template,
        land_type=cfg.data.land_type,
    )

    # Load template and fire data
    template = prepare_template(cfg.data.template)
    mtbs = gpd.read_file(cfg.data.mtbs).to_crs(cfg.epsg)

    # Merge it to get years
    mtbs["year"] = mtbs.Ig_Date.dt.year

    # Prepare results
    results = pd.read_csv(cfg.data.att_results)
    results["land_type"] = 2.0  # This is a bit hardcoded
    results["land_type"] = results.land_type.replace(
        cfg.simulations.land_type_mapping
    )

    spillover_results = pd.read_csv(cfg.data.spillover_att_results)
    spillover_results = (
        spillover_results[
            spillover_results.dist_treat == cfg.simulations.dist_treat
        ]
        .drop(columns="dist_treat", errors="ignore")
        .sort_values("year")
    )

    if cfg.simulations.year_limit is not None:
        spillover_results = spillover_results[
            spillover_results.year <= cfg.simulations.year_limit
        ]

    sim_data = sim_data[sim_data.land_type.isin(cfg.simulations.land_type)]

    # Prepare fire data
    fire_data = pd.read_feather(cfg.data.fire_treatments).drop(
        columns=["spatial_ref"], errors="ignore"
    )
    fire_data = fire_data[fire_data.year >= cfg.simulations.start_year]

    # Prepare template
    template = prepare_template(
        cfg.data.template, years=cfg.simulations.template_years
    )
    template = template[template.year >= cfg.simulations.start_year]

    # Iterate over size of treatments
    for i in cfg.simulations.treatment_sizes:
        log.info(f"Running simulations for {i} size treatment")
        run_simulations(
            template=template,
            sim_data=sim_data,
            results=results,
            fire_data=fire_data,
            save_path=f"{cfg.data.sims_save_path}_{i}",
            num_sims=cfg.simulations.num_sims,
            step_save=cfg.simulations.step_save,
            size_treatment=cfg.simulations.size_treatment,
            start_year=cfg.simulations.start_year,
            sample_n=i,
            crs=cfg.epsg,
            format=cfg.simulations.format,
            spillovers=cfg.simulations.spillovers,
            spillover_size=cfg.simulations.spillover_size,
            spillover_estimates=spillover_results,
        )


if __name__ == "__main__":
    main()
