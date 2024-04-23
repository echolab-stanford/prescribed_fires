import logging
import os

import duckdb
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def tyra(path_to_db, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save best models by both loss and asmd
    best_model_loss = duckdb.query(
        f"""
        with models_loss_min as(
        select model_run_id, focal_year, min(loss) as min_loss
        from '{path_to_db}/loss/*.parquet'
        group by model_run_id, focal_year
        order by focal_year
        ), min_loss_year as (
        select focal_year, min(min_loss) as min_min_loss
        from models_loss_min
        group by focal_year
        ) select a.focal_year, a.model_run_id, b.min_min_loss
        from models_loss_min a
        join min_loss_year b
        on a.focal_year = b.focal_year and a.min_loss = b.min_min_loss
        """
    ).to_df()

    best_model_loss.to_csv(os.path.join(save_path, "best_model_loss.csv"), index=False)

    best_model_asmd = duckdb.query(
        f"""
        WITH averages_model_year AS (
        SELECT model_run_id, focal_year, avg(std_weighted_asmd) as mean_std_weighted_asmd
        FROM '{path_to_db}/std_diffs/*.parquet'
        GROUP BY model_run_id, focal_year
        ), min_asmd_per_year as (
            SELECT focal_year,  min(mean_std_weighted_asmd) as min_std_weighted_asmd
            FROM averages_model_year
            GROUP BY focal_year
        )
        SELECT a.focal_year, a.model_run_id, m.min_std_weighted_asmd
        FROM averages_model_year a
        JOIN min_asmd_per_year m 
        ON a.focal_year = m.focal_year AND 
        a.mean_std_weighted_asmd = m.min_std_weighted_asmd
        order by a.focal_year
        """
    ).to_df()

    best_model_asmd.to_csv(os.path.join(save_path, "best_model_asmd.csv"), index=False)

    # Save all stadarized differences for the best model (loss and asmd)
    std_diffs_loss = duckdb.query(
        f"""
    SELECT *
    FROM '{path_to_db}/std_diffs/*.parquet'
    WHERE model_run_id IN {repr(tuple(map(str, best_model_loss.model_run_id.tolist())))}
    """
    ).to_df()

    std_diffs_loss.to_csv(
        os.path.join(save_path, "best_model_loss_std_diffs.csv"), index=False
    )

    std_diffs_asmd = duckdb.query(
        f"""
    SELECT *
    FROM '{path_to_db}/std_diffs/*.parquet'
    WHERE model_run_id IN {repr(tuple(map(str, best_model_loss.model_run_id.tolist())))}
    """
    ).to_df()

    std_diffs_asmd.to_csv(
        os.path.join(save_path, "best_model_asmd_std_diffs.csv"), index=False
    )

    # Save all weights for the best model (loss and asmd)

    weights = duckdb.query(
        f"""
    SELECT weights, focal_year, row_id as grid_id
    FROM '{path_to_db}/results/*.parquet'
    WHERE model_run_id IN  {repr(tuple(map(str, best_model_asmd.model_run_id.tolist())))}
    """
    ).to_df()

    weights.to_csv(os.path.join(save_path, "best_model_loss_weights.csv"), index=False)

    weights = duckdb.query(
        f"""
    SELECT weights, focal_year, row_id as grid_id
    FROM '{path_to_db}/results/*.parquet'
    WHERE model_run_id IN  {repr(tuple(map(str, best_model_asmd.model_run_id.tolist())))}
    """
    ).to_df()

    weights.to_csv(os.path.join(save_path, "best_model_asmd_weights.csv"), index=False)

    return None


@hydra.main(config_path="../conf", config_name="tyra")
def run_tyra(cfg: DictConfig) -> None:
    for key, path in cfg.estimate.path_results.items():
        log.info(f"Processing {key}")

        save_path_conf = os.path.join(cfg.save_path, key)

        tyra(save_path=save_path_conf, path_to_db=path)


if __name__ == "__main__":
    run_tyra()
