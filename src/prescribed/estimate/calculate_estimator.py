import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
    scale: Optional[float] = False,
    pooling: Optional[bool] = False,
    **kwargs,
) -> pd.DataFrame:
    """Calculate lagged effects for a a given focal year

    This function calculates the estimator for each of the lagged years in our data using a weighted average. If the `pooling` option is passed, then we pool the estimates across the lags using a OLS model with Jackknife errors (for more details see `pooling_estimates` function).

    See parameters to understand the rest of the options.

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
    scale : Optional[float]
        A scaling factor for the estimator. Default is False
    pooling : Optional[bool]
        If we want to pool the estimators across lags. Default is False
    **kwargs
        Additional arguments to pass to the `pooling_estimates` function

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
        lags_arr = np.arange(focal_year + 1, lag_up_to)
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

        # Scale the estimator
        if scale:
            estimator.loc[:, "delta_dnbr"] = estimator["delta_dnbr"] * scale
            estimator.loc[:, "delta_frp"] = estimator["delta_frp"] * scale

        results.append(estimator)

    df = pd.concat(results)

    # Pool the estimates if passes
    if pooling:
        df = pooling_estimates(df, **kwargs)

    return df


def pooling_estimates(df: pd.DataFrame, cluster_var: str, formula: str) -> pd.DataFrame:
    """Calculate variance/ses for pooled estimators

    This function pools the estimates to calculate the variance of the relationship between the ATTs and the lags. To do this we will use a simple OLS fit and to estimate the variance just use a LOO (leave-one-out) approach as a response of the SC design (i.e. control and treatment groups can have overlaps over time in consecutive years). This function will take results from the `calculate_estimator` (and actually be inside this function if the user want SEs), thus the input should be the output of the `calculate_estimator` function.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe with the estimator for each of the lagged years
    cluster_var : str
        The name of the cluster variable for the Jackkife estimation
    fomula : str
        The formula to use in the OLS estimation. This is any valid formula for `patsy` (thus anything you can pass to statsmodels.smf.api)

    Returns
    -------
    pd.DataFrame
        A dataframe with the variance for each of the lagged years
    """

    # Build year samples for cluster-year Jackknife
    cluster = df[cluster_var].unique()
    lags = df["lag"].unique()

    # Create jk samples: loo for year year
    lists = [
        [elem for j, elem in enumerate(cluster) if j != i] for i in range(len(cluster))
    ]

    # Mean model with all the sample
    model_all_years = smf.ols(formula=formula, data=df)
    result_all_years = model_all_years.fit()
    coefs_all_years = result_all_years.predict(pd.DataFrame({"lag": lags}))

    # Model using OLS for each Jackknife sample and store in nd-array (samples, lags)
    arr = np.zeros((len(lists), lags.size))  # Fill array of sample / lags
    for jk_sample in lists:
        model = smf.ols(
            formula=formula,
            data=df[df[cluster_var].isin(jk_sample)],
        )
        result = model.fit()

        # Calculate predictions for each lag
        lag_df = pd.DataFrame({"lag": lags})
        coefs = result.predict(lag_df)

        arr[lists.index(jk_sample), :] = coefs.values

    # Calculate the standard errors for each jk sample
    ses = np.apply_along_axis(
        lambda x: np.sqrt(np.var(x) * (x.size - 1) ** 2 / x.size), 0, arr
    )

    # Calculate the CIs
    low_ci, high_ci = (
        coefs_all_years.values - 1.96 * ses,
        coefs_all_years.values + 1.96 * ses,
    )

    # Store the results
    results = pd.DataFrame(
        {
            "year": lags,
            "coef": coefs_all_years,
            "low_ci": low_ci,
            "high_ci": high_ci,
            "se": ses,
        }
    )

    return results
