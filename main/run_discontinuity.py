import pandas as pd
import geopandas as gpd
import rioxarray
import hydra
import logging
from omegaconf import DictConfig
from prescribed.estimate.calculate_rdd import rdds_along_distances

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="discontinuity")
def run_discontinuity_grid(cfg: DictConfig) -> tuple:
    mtbs = gpd.read_file(cfg.mtbs_path)

    mtbs_ca = mtbs[
        (mtbs.Event_ID.str.contains("CA"))
        & (mtbs.Incid_Type.isin(["Wildfire", "Prescribed Fire"]))
    ].to_crs("3310")
    mtbs_ca["Ig_Date"] = pd.to_datetime(mtbs_ca.Ig_Date)
    mtbs_ca["year"] = mtbs_ca.Ig_Date.dt.year

    # Load California boundaries
    ca = gpd.read_file(cfg.mask).to_crs("4326")

    template = rioxarray.open_rasterio(cfg.template)

    # Merge with dnbr data to get outcomes
    dnbr = pd.read_feather(cfg.outputs.dnbr)

    dnbr = dnbr.merge(
        mtbs_ca[["Event_ID", "year"]],
        left_on="event_id",
        right_on="Event_ID",
        how="left",
    ).drop(columns=["lat", "lon", "event_id"])

    dist_results_dnbr = rdds_along_distances(
        outcome_df=dnbr,
        distances=cfg.distances,
        rdd_kws={
            "lags": range(1, 12),
            "years": range(2000, 2021),
            "outcome_var": "dnbr",
            "id_var": "grid_id",
            "plot_rdd": False,
        },
        dict_dist_kws={
            "mtbs_shapefile": mtbs_ca,
            "template": template,
            "pop_threshold": cfg.population.pop_tresh,
            "pop_raster_path": cfg.population.pop_raster_path,
            "mask": ca,
        },
        save_path=f"./rdd_results_dnbr_{cfg.distance}.csv",
    )

    # Merge with dnbr data to get outcomes
    frp = pd.read_feather(cfg.outputs.frp)

    dist_results_frp = rdds_along_distances(
        outcome_df=frp,
        distances=cfg.distances,
        rdd_kws={
            "lags": range(1, 12),
            "years": range(2000, 2021),
            "outcome_var": "cum_frp",
            "id_var": "grid_id",
            "plot_rdd": False,
        },
        dict_dist_kws={
            "mtbs_shapefile": mtbs_ca,
            "template": template,
            "pop_threshold": cfg.population.pop_thresh,
            "pop_raster_path": cfg.population.pop_raster_path,
            "mask": ca,
        },
        save_path=f"./rdd_results_frp_{cfg.distance}.csv",
    )

    return dist_results_dnbr, dist_results_frp


if __name__ == "__main__":
    run_discontinuity_grid()
