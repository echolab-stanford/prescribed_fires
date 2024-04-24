import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)


def calculate_estimator(
    treatments: pd.DataFrame,
    weights: pd.DataFrame,
    outcomes: pd.DataFrame,
    focal_year: int,
    outcome_var: str,
    low_treatment_class: list = [2, 1],
    lag_up_to: int = 2021,
) -> pd.DataFrame:
    """Calculate lagged effects for a a given focal year

    This function calculates the estimator for each of the lagged years in our data using a weighted average.

    Parameters
    ----------
    treatments : pd.DataFrame
        A dataframe with the treatments and intensities data using the FRP and the DNBR. This are the same wide treatments that we created during balancing (see `prescribed.main.build_dataset`) which created the needed file for this function.
    outcomes : pd.DataFrame
        A dataframe with the target outcomes to calculate the estimator. This should be a `pd.DataFrame` with dependent variables with a grid identifier and a year.
    weights : pd.DataFrame
        A dataframe with the weights data from the balancing routine.
    focal_year : int
        The focal year for which we want to calculate the estimator
    outcome_var : str
        The name of the outcome variable in the `outcomes` dataframe
    low_treatment_class : list
        The class of the low treatment for [dnbr, frp]. Default is [2, 1]
    lag_up_to : int
        The maximum lag that we want to calculate the estimator

    Returns
    -------
    pd.DataFrame
        A dataframe with the estimator for each of the lagged years
    """

    column_names = ["treat", "class_dnbr", "class_frp"]

    results = []
    for focal_year in tqdm(focal_year, desc="Calculating estimator per focal year..."):
        # Filter the treatments for the focal year
        columns_names_year = [f"{c}_{focal_year}" for c in column_names]
        treat_name, dnbr_class, frp_class = columns_names_year

        treatments_year = treatments[["grid_id"] + columns_names_year]

        # Get the means for the treatment grids
        treat_means = []
        for column_treat, low_class in zip(
            [dnbr_class, frp_class], low_treatment_class
        ):
            # Define treatment: exposed pixels * pixel intensity/severity
            treatments_year.loc[:, "treat"] = (
                treatments_year[treat_name] * treatments_year[column_treat]
            )
            treatments_year.loc[:, "treat"] = np.where(
                treatments_year["treat"] == low_class, 1, 0
            )
            treatments_year = treatments_year[treatments_year.treat == 1]

            # Get means for treatment grids
            outcomes_year = outcomes[
                outcomes.grid_id.isin(treatments_year.grid_id.tolist())
            ]
            treat_means.append(outcomes_year.groupby("year")[outcome_var].mean())

        # Now, loop over the weights per year and calculate the means for each year outcomes
        control_means_list = []
        lags_arr = np.arange(focal_year, lag_up_to)
        for year in lags_arr:
            weights_year = weights[weights.focal_year == year]
            outcomes_control = outcomes[
                outcomes.grid_id.isin(weights_year.grid_id.tolist())
            ]

            # Merge to aligned dataset
            outcomes_control = outcomes_control.merge(weights_year, on="grid_id")

            # Calculate the weighted average
            avg_c = np.average(
                outcomes_control[outcome_var].values,
                weights=outcomes_control.weights.values,
            )
            control_means_list.append(avg_c)

        control_means = pd.DataFrame(
            {
                "year": lags_arr,
                "control_means": control_means_list,
                "lag": lags_arr - focal_year,
                "focal_year": focal_year,
            }
        )

        # Concatenate the results
        treat_means = pd.concat(treat_means, axis=1).reset_index()
        treat_means.columns = ["year", "dnbr", "frp"]
        estimator = treat_means.merge(control_means, on="year")

        # Calculate esimator for both treatments
        estimator.loc[:, "delta_dnbr"] = estimator["dnbr"] - estimator["control_means"]
        estimator.loc[:, "delta_frp"] = estimator["frp"] - estimator["control_means"]
        results.append(estimator)

    df = pd.concat(results)

    return df
