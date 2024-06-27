import os
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from prescribed.utils import tqdm_joblib
from tqdm import tqdm

from .create_distances import create_distances
from .rdrobust import rdplot, rdrobust


def estimation_rdd(
    year: int,
    lag: int,
    data: pd.DataFrame,
    outcome_var: int,
    id_var="grid_id",
    plot_rdd=False,
) -> pd.DataFrame:
    """Regression discontinuity estimation for a given year and lag.

    Estimate a regression discontinuity using a panel dataset. The RDD is spatial,
    this function expects that data has a "distance" column that represents the
    discontinuity (centered in zero) and an id_var to merge the data.

    The regression will estimate the effects in time, so will use distance in time t
    to estimate the effect in time t+1  (assuming lag=1), and t+2 (lag=2), etc.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with distance, outcome_var and id_var columns.
    year : int
        Year of the treatment.
    lag : int
        Lag to estimate the effect of the treatment.
    outcome_var : str
        Outcome variable to estimate the RDD.
    id_var : str, optional
        Identifier variable to merge the data, by default "grid_id".
    plot_rdd : bool, optional
        Plot the RDD estimation, by default False.

    Returns
    -------
    pd.DataFrame
        RDD estimation results as a dataframe. The dataframe will have the following columns:
        - coef: Estimated coefficient
        - ci_low: Confidence interval
        - ci_high: Confidence interval
        - year: Year of the treatment
        - lag: Lag of the effect
        - bw: Bandwidth estimation
    """
    year_treat = year
    year_outcome = year + lag

    # Separate data to get consistent treatment and outcome years
    running_treat = data[data.year == year_treat][[id_var, "distance"]]
    outcome = data[data.year == year_outcome][[id_var, outcome_var]]

    # Fill in missing values for dnbr. Assume that if no measure, we have zero.
    outcome.loc[outcome[outcome_var] == 0, outcome_var] = np.nan
    outcome[outcome_var] = outcome[outcome_var].fillna(0)

    # Merge data for plotting
    data_reg = running_treat.merge(outcome, on=id_var, how="outer")

    # Running RDD esimation using Calonico, et.al., (2014) estimation
    try:
        est = rdrobust(
            y=data_reg[outcome_var].values, x=data_reg.distance.values, c=0, all=True
        )
        est_res = pd.concat([est.coef, est.ci], axis=1)
        est_res["year"] = year_treat
        est_res["lag"] = lag
        est_res["bw"] = est.bws["left"].values[0]

        if plot_rdd:
            # Subset to the bandwidth estimation only (the rest is not valid in the RDD context)
            h_l, h_r = est.bws.loc["h", :].values
            subset = (-h_l <= data_reg.distance.values) & (
                data_reg.distance.values <= h_r
            )

            # RD-plot with 95% confidence intervals
            _, rd_bins, rd_poly = rdplot(
                y=data_reg[outcome_var].values,
                x=data_reg.distance.values,
                subset=subset,
                kernel="triangular",
                h=[h_l, h_r],
                ci=95,
                title="",
                y_label=r"$dNBR$ at time $t+1$",
                x_label=r"Distance to wildfire boundary in time $t$",
                plot=False,
            )

            # Add year of treatment to the plot
            rd_poly["year"] = year_treat
            rd_bins["year"] = year_treat

        # Rename the columns to get sensible names
        est_res = est_res.rename(
            columns={"Coeff": "coef", "CI Lower": "ci_low", "CI Upper": "ci_high"}
        )

    except Exception as e:
        pass
        print(f"Cannot estimate RDD for year/lag: {year}/{lag}: {e}")

        est_res, rd_poly, rd_bins = None, None, None

    # Define output
    if plot_rdd:
        return est_res, rd_bins, rd_poly
    else:
        return est_res


def parallel_run_estimation_rdd(lags: list, years: list, **kwargs) -> pd.DataFrame:
    """Run RDD estimation for multiple years and lags in parallel

    Parameters
    ----------
    lags : list
        List of lags to estimate the RDD effect.
    years : list
        List of years to estimate the RDD effect.
    kwargs : dict
        Additional arguments to pass to `estimation_rdd`.
    """

    # Run RDD estimation in parallel
    with tqdm_joblib(
        tqdm(desc="RDD calculation", total=len(lags) * len(years))
    ) as progress_bar:
        results = Parallel(n_jobs=18)(
            delayed(estimation_rdd)(year, lag, **kwargs)
            for year, lag in product(years, lags)
        )

    # Concatenate results
    if isinstance(results[0], tuple):
        est_res, rd_bins, rd_poly = zip(*results)
        est_res = pd.concat(est_res)
        rd_bins = pd.concat(rd_bins)
        rd_poly = pd.concat(rd_poly)

        return est_res, rd_bins, rd_poly
    else:
        est_res = pd.concat(results)

        return est_res


def rdds_along_distances(
    outcome_df: pd.DataFrame,
    distances: list,
    rdd_kws: dict,
    dict_dist_kws: dict,
    save_dir: str = None,
    file_name: str = None,
) -> pd.DataFrame:
    """Run multiple RDD estimations for different lags, years, and distances"""

    # Check buffers in case they're not lists, we don't want the loop to fail
    if not isinstance(distances, list):
        distances = [distances]

    # Loop and store!
    list_dfs = []
    for buf in distances:
        distance_df = create_distances(
            mtbs_shapefile=dict_dist_kws["mtbs_shapefile"],
            template=dict_dist_kws["template"],
            pop_threshold=dict_dist_kws["pop_threshold"],
            buffer_treatment=buf,
            buffer=buf * 2,
            pop_raster_path=dict_dist_kws["pop_raster_path"],
            mask=dict_dist_kws["mask"],
        )

        distance_df = distance_df.dropna(subset="grid_id")

        # Merge with dnbr data to get outcomes
        distance_df = distance_df.merge(
            outcome_df, on=[rdd_kws["id_var"], "year"], how="left"
        )

        # Run RDD estimation in parallel for all years/lags
        est_res = parallel_run_estimation_rdd(
            lags=rdd_kws["lags"],
            years=rdd_kws["years"],
            data=distance_df,
            outcome_var=rdd_kws["outcome_var"],
            id_var=rdd_kws["id_var"],
            plot_rdd=False,
        )

        # Create a dummy var if Coeff is significant using the CIs
        est_res["insignificant"] = (est_res["ci_low"] <= 0) & (est_res["ci_high"] > 0)

        est_res["treat_buffer"] = buf
        list_dfs.append(est_res)

    if save_dir & file_name:
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir), exist_ok=True)

        save_path = os.path.join(save_dir, file_name)
        df = pd.concat(list_dfs).reset_index()
        df = df.rename(columns={"index": "est_type"})
        df.to_csv(save_path, index=False)

    return pd.concat(list_dfs)
