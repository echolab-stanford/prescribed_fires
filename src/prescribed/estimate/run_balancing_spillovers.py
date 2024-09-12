import logging
import os
import re
from itertools import product
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from ..utils import generate_run_id
from .cbps_torch import CBPS

log = logging.getLogger(__name__)


def safe_makedirs(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def run_balancing_spillovers(
    df: pd.DataFrame,
    focal_year: int,
    row_id: str,
    reg_list: list,
    lr_list: list,
    metrics: Union[str, List[str]],
    save_path: str,
    distance_treatment: int = 5_000,
    extra_dict_elements: dict = None,
    **kwargs,
) -> None:
    """Run balancing for a given focal year using CBPS for spillover treatments

    This function runs the CBPS balancing algorithm to quatify spillover effects
    for a given focal year and saves the weights for analysis. The function does
    not work like the `run_balancing` function as we are not interested in the
    severity/intensity, but in general effects. Thus we don't use a wide-format
    dataframe, but a long-format dataframe with distances and potential treated
    units.

    We want to do balancing after a focal year for a given treatment year. To do
    this we will keep all columns that exist before the focal year up to the treatment
    year. This will allow us to balance the data for the treatment year using the
    data available before the treatment year.

    Other arguments can be passed to the CBPS function using the kwargs argument.

    Parameters
    -----------
     df: pd.DataFrame
        A long dataframe with all columns correctly labeled, the function will
        look for the `distance` column to subset the treatments.
    focal_year: int
        The year to be used as the focal year
    row_id: str
        The name of the column to be used as row id
    reg_list: list
        A list with the regularization parameters to be used
    lr_list: list
        A list with the learning rate parameters to be used
    metrics: Union[str, List[str]]
        The method to be used for assess balace in standarized differences
    save_path: str
        The path to save the results
    distance_treatment: float
        The distance from the wildfire to be considered as a treatment. It must
        be an integer value in meters.
    extra_dict_elements: dict
        Extra elements to be added to all the results dataframes. This is useful
        if you want ot pass metadata. Default is None

    Returns
    --------
    None
    """

    # Define values that are not missing in the distance column
    w = np.where(df.distance.values <= distance_treatment, 1, 0)

    # Test that we have treatments!
    if w.sum() == 0:
        raise ValueError("No treatments found in the data!")

    # Row id using grid_id to merge back with controls at the end
    id_col = df[row_id].values
    id_col = id_col[w == 0]

    # Drop grid_id and treatment and some columns that we don't need!
    df = df.drop(
        columns=[row_id, "distance", "donut", "wildfire", "lat", "lon"],
        errors="ignore",
    )

    # Select columns to drop for balancing (all after the focal year and the min year)
    cols_keep = [
        col
        for col in df.columns.tolist()
        if re.search(r"\d+", col)
        and 2000 < int(re.findall(r"\d+", col)[0]) < focal_year
    ]

    # Add columns without digits, these are the covariates that are stable in time
    cols_keep += [col for col in df.columns if not re.search(r"\d+", col)]

    # Keep columns before focal year
    df = df[cols_keep]

    # Scale data
    X = MinMaxScaler().fit_transform(df.values)

    # Run CBPS for all combinations of regularization and learning rate
    for reg, lr in product(reg_list, lr_list):
        # Generate hash id for model run
        run_id = generate_run_id([reg, lr, focal_year])

        # Run balancing
        cbps = CBPS(X=X, W=w, estimand="ATT", reg=reg, lr=lr, **kwargs)
        weights = cbps.weights(numpy=True)

        # Save results as a dataframe
        df_results = pd.DataFrame({"weights": weights}, dtype=(np.float64))

        df_results = df_results.assign(
            reg=reg,
            lr=lr,
            row_id=id_col,
            model_run_id=run_id,
            focal_year=focal_year,
        )
        df_results = df_results.assign(**extra_dict_elements)

        # Save standarized differences for all metrics
        list_metrics = []
        for metric in metrics:
            std_diffs = cbps._covariate_differences(metric=metric, device="cpu")
            std_diffs_df = pd.concat(std_diffs, axis=1)
            std_diffs_df.columns = [
                f"std_unweighted_{metric}",
                f"std_weighted_{metric}",
            ]
            list_metrics.append(std_diffs_df)

        # Merge all metrics and add covariate name
        std_diffs_df = pd.concat(list_metrics, axis=1)

        std_diffs_df = std_diffs_df.assign(
            reg=reg,
            lr=lr,
            model_run_id=run_id,
            focal_year=focal_year,
        )
        std_diffs_df = std_diffs_df.assign(**extra_dict_elements)

        # If cbps has an intercept, at it to the column name
        if cbps.intercept:
            cols = ["Intercept"] + df.columns.tolist()
        else:
            cols = df.columns.tolist()
        std_diffs_df["covar"] = cols

        # Save loss to a dataframe
        df_loss = pd.DataFrame(
            {
                "loss": cbps.loss,
                "lr_decay": np.array(cbps.lr_decay),
                "iter": cbps.niter,
                "niter": range(1, cbps.niter + 1),
            }
        )
        df_loss = df_loss.assign(
            reg=reg,
            lr=lr,
            model_run_id=run_id,
            focal_year=focal_year,
        )

        if extra_dict_elements is not None:
            df_loss = df_loss.assign(**extra_dict_elements)

        if save_path:
            if not os.path.exists(save_path):
                safe_makedirs(save_path)

                # Create intermediary folders
                safe_makedirs(os.path.join(save_path, "results"))
                safe_makedirs(os.path.join(save_path, "std_diffs"))
                safe_makedirs(os.path.join(save_path, "loss"))

            df_results.to_parquet(
                os.path.join(save_path, "results", f"results_{run_id}.parquet")
            )
            std_diffs_df.to_parquet(
                os.path.join(
                    save_path, "std_diffs", f"std_diffs_{run_id}.parquet"
                )
            )
            df_loss.to_parquet(
                os.path.join(save_path, "loss", f"loss_{run_id}.parquet")
            )

            log.info(f"Finish loading data in {save_path}")
        # Clean memory
        del cbps, weights, df_results, std_diffs_df, df_loss
        torch.cuda.empty_cache()

    return None
