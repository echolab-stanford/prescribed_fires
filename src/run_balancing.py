import os
import re
from itertools import product

import numpy as np
import pandas as pd
import torch
from .cbps_torch import CBPS
from sklearn.preprocessing import MinMaxScaler


def run_balancing(
    df,
    focal_year,
    treat_col,
    class_col,
    row_id,
    reg_list,
    lr_list,
    metrics,
    save_path,
    intensity_class=1,
    **kwargs,
):
    """Run balancing for a given focal year using CBPS.

    This function run the CBPS balancing algorithm for a given focal year and
    saves the weights for analysis. The function does rely on a wide dataframe
    with all columns correctly labeled as "{var}_{year}". If more than one number
    is in the column name, the function will select the first and use this to
    filter the dataframe columns.

    We want to do balancing after a focal year for a given treatment year. To do
    this we will keep all columns that exist before the focal year up to the treatment
    year. This will allow us to balance the data for the treatment year using the
    data available before the treatment year.

    Other arguments can be passed to the CBPS function using the kwargs argument.

    Args:
        df (pd.DataFrame): A wide dataframe with all columns correctly labeled
        focal_year (int): The year to be used as the f  ocal year
        treat_col (str): The name of the treatment column
        row_id (str): The name of the column to be used as row id
        reg_list (list): A list with the regularization parameters to be used
        lr_list (list): A list with the learning rate parameters to be used
        metrics (str ot list): The method to be used for assess balace in standarized differences
        save_path (str): If True, the function will save the results to a file
        intesnity_class (int): The intensity/severity class to use to define control
        **kwargs: Other arguments to be passed to the CBPS function

    Returns:
        None
    """

    # Define treatment with treatment column and class column
    treat_class = (df[treat_col] * df[class_col]).values
    w = np.where(treat_class >= intensity_class, 1, 0)

    # Row id using grid_id
    id_col = df[row_id].values
    id_col = id_col[w == 0]

    # Drop grid_id and treatment
    df = df.drop(columns=[row_id, treat_col])

    # Select columns to drop for balancing (all before the focal year)
    cols_keep = [
        col
        for col in df.columns.tolist()
        if re.search(r"\d+", col)
        and 2000 < int(re.findall(r"\d+", col)[0]) < focal_year
    ]

    # Add columns without digists, these are the covariates that are stable in time
    cols_keep += [col for col in df.columns if not re.search(r"\d+", col)]

    # Keep columns before focal year
    df = df[cols_keep]

    # Drop some columns that we don't need!
    df = df.drop(columns=[col for col in df.columns if "treat" in col])

    # Scale data
    X = MinMaxScaler().fit_transform(df.values)

    # Run CBPS for all combinations of regularization and learning rate
    for reg, lr in product(reg_list, lr_list):
        cbps = CBPS(X=X, W=w, estimand="ATT", reg=reg, lr=lr, **kwargs)
        weights = cbps.weights(numpy=True)

        # Save results as a dataframe
        df_results = pd.DataFrame(weights)
        df_results.columns = [f"weights_{cbps.reg}_{cbps.lr}"]

        # Add row id
        df_results[row_id] = id_col

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

        # If cbps has an intercept, at it to the column name
        if cbps.intercept:
            cols = ["Intercept"] + df.columns.tolist()
        else:
            cols = df.columns.tolist()
        std_diffs_df["covar"] = cols

        # Save loss to a dataframe
        df_loss = pd.DataFrame(
            {"loss": cbps.loss, "lr": np.array(cbps.lr_decay), "iter": cbps.niter}
        )

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            df_results.to_feather(
                f"{save_path}/weights_{focal_year}_{lr}_{reg}.feather"
            )
            std_diffs_df.to_feather(
                f"{save_path}/std_diffs_{focal_year}_{lr}_{reg}.feather"
            )
            df_loss.to_feather(f"{save_path}/loss_{focal_year}_{lr}_{reg}.feather")

        # Clean memory
        del cbps, weights, df_results, std_diffs_df
        torch.cuda.empty_cache()

    return None
