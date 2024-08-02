"""Run balancing for a focal year and a specific land type

This script is a wrapper for the run_balancing function. It prepares the data and runs the balancing for a specific land type.

Land-types are defined as a integer:
    1: "Agricultural",
    - 2: "Conifer",
    - 3: "Conifer-Hardwood",
    4: "Developed",
    5: "Exotic Herbaceous",
    6: "Exotic,Tree-Shrub",
    7: "Grassland",
    - 8: "Hardwood",
    9: "No Data",
    10: "Non-vegetated",
    11: "Riparian",
     - 12: "Shrubland",
    13: "Sparsely Vegetated"
"""

import logging
from functools import reduce

import hydra
import pandas as pd
from omegaconf import DictConfig
from prescribed.estimate.run_balancing_spillovers import run_balancing_spillovers

log = logging.getLogger(__name__)


def prepare_data(dict_paths, treats_long, focal_year):
    # Load treats
    if isinstance(treats_long, str):
        treats_long = pd.read_feather(treats_long)

    # Filter by year
    treats_long = treats_long[treats_long.year == focal_year]

    # Drop year column from treatment
    treats_long.drop(columns="year", inplace=True)

    # Load datasets and merge with template
    data = []
    for key, path in dict_paths.items():
        print(key)
        df = pd.read_feather(path)

        # Remove columns if present
        if "index" in df.columns:
            df.drop(columns="index", inplace=True)

        if "lat" in df.columns:
            df.drop(
                columns=[c for c in df.columns if "lat" in c or "lon" in c],
                inplace=True,
            )

        data.append(df)

    # Merge all datasets
    df = reduce(lambda x, y: x.merge(y, on="grid_id", how="left"), data)

    # Drop missing values
    df = df.dropna()

    # Merge with treatments
    df = df.merge(treats_long, on="grid_id")

    # Save some memory
    del data

    return df


@hydra.main(version_base=None, config_path="../conf", config_name="spillovers")
def main(cfg: DictConfig) -> None:
    log.info("Building dataset")
    # Prepare data
    df = prepare_data(
        dict_paths=cfg.data.dict_paths,
        treats_long=cfg.data.treats_long,
        focal_year=cfg.balancing.focal_year,
    )

    # Run balancing
    run_balancing_spillovers(
        df=df.dropna(),
        focal_year=cfg.balancing.focal_year,
        treat_col=f"treat_{cfg.balancing.focal_year}",
        class_col=f"class_{cfg.balancing.treatment}_{cfg.balancing.focal_year}",
        row_id=cfg.balancing.row_id,
        reg_list=cfg.balancing.reg,
        lr_list=cfg.balancing.lr,
        intercept=cfg.balancing.intercept,
        niter=cfg.balancing.niter,
        metrics=cfg.balancing.metrics,
        save_path=cfg.balancing.save_path,
        distance_treatment=cfg.balancing.distance_treatment,
    )

    log.info("Finish with balancing. All files are saved and CV is done!")


if __name__ == "__main__":
    main()
