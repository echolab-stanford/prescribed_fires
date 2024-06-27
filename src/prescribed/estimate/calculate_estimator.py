import logging
import re
from typing import Optional, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tqdm

log = logging.getLogger(__name__)


def calculate_estimator(
    treatments: pd.DataFrame,
    weights: pd.DataFrame,
    outcomes: Union[pd.DataFrame, str],
    focal_year: int,
    outcome_var: str,
    low_treatment_class: dict = {"dnbr": 1},
    high_class: int = 3,
    scale: Optional[float] = False,
    pooling: Optional[bool] = False,
    **kwargs,
) -> pd.DataFrame:
    """Calculate lagged effects for a a given focal year

    This function calculates the estimator for each of the lagged years in our data using a weighted average. If the `pooling` option is passed, then we pool the estimates across the lags using a OLS model with Jackknife errors (for more details see `pooling_estimates` function).

    The estimator is calculated for either a ATT using a dataset with an outcome variable (`outcome_var`) or a RR estimator using the grid ids of the treatment and control groups. If `rr` is passed, the latter estimator will be used and the `outcomes` variable will be ignored. These RR estimator is the ratio of the number of treatment pixels at non-low severity and the weighted number of pixels in the synth control group (notice this number can be a float: 19.6 pixels exposed, for example).

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
    low_treatment_class : dict
        The low treatment definition. The key must be the variable in the treatment dataset and the value should be the class number. The default is `{"dnbr": 1}`.
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

    # Column names stubs on the default treatment dataset
    column_names = ["treat", "class_dnbr", "class_frp"]

    # Check the low treatment class definition
    if len(low_treatment_class) > 1:
        raise ValueError("Only one treatment class can be defined")

    treat_class, low_class = list(low_treatment_class.items())[0]

    results = []
    for focal_year in tqdm(focal_year, desc="Calculating estimator per focal year..."):
        # We want to calculate an ATT estimator per each focal year and lag,
        # these are  the next available years after the focal year in the
        # outcomes dataset (or in the treatments dataset if we are calculating
        # the RR estimator). To do this we will first calculate the treatment
        # means, these are just raw means of all the treated pixels according to
        # the low-severity/intensity classification.
        columns_names_year = [f"{c}_{focal_year}" for c in column_names]
        treat_name, dnbr_class, frp_class = columns_names_year

        treatments_year = treatments[["grid_id"] + columns_names_year]

        # Define the treatment column and the treatment class column
        column_treat = dnbr_class if treat_class == "dnbr" else frp_class

        # Define treatment: exposed pixels * pixel intensity/severity for a
        # given focal year
        treatments_year.loc[:, "treat"] = (
            treatments_year[treat_name] * treatments_year[column_treat]
        )
        # Define treatment as the low class
        treatments_year.loc[:, "treat_class"] = np.where(
            treatments_year["treat"] == low_class, 1, 0
        )

        # Get means for treatment grids after focal year (notice here we do
        # include the focal year as we want to calculate the ATT for the
        # focal year, this is not true for the RR)
        if isinstance(outcomes, pd.DataFrame):
            # Filter to get only the treated ones mean value
            treatments_year = treatments_year[treatments_year["treat_class"] == 1]
            outcomes_year = outcomes[
                outcomes.grid_id.isin(treatments_year.grid_id.tolist())
                & (outcomes.year >= focal_year)
            ]

            # Calculate the raw mean per each year and create a dataframe
            treat_means_df = outcomes_year.groupby("year", as_index=False)[
                outcome_var
            ].mean()

            # Rename grouped var to something more meaningful
            treat_means_df.rename(columns={outcome_var: "treat_mean"}, inplace=True)

        elif outcomes == "rr":
            # We want to build the numerator of the RR estimator here. This
            # is the total number of observations in the treatment group that
            # are not low severity in the next periods (here we care about
            # the next periods as is the ATT, thus the effect on the focal
            # year (lag = 0) is not of interest).

            # Reformat the column_treat to be compatible in the long format of
            # the dataframe: for example, `class_dnbr_2010` to `class_dnbr`
            column_treat = re.sub(r"_\d+", "", column_treat)

            # Get treated grids in focal year
            treat_sample = treatments_year[
                treatments_year["treat_class"] == 1
            ].grid_id.tolist()

            # Reshape data to make the subsetting easier and to calculate
            # counts across time with ease.
            treatments_sample = treatments[treatments.grid_id.isin(treat_sample)]
            treatments_sample_long = pd.wide_to_long(
                treatments_sample,
                stubnames=column_names,
                i="grid_id",
                j="year",
                sep="_",
            ).reset_index()

            # Remove past treatment years and keep only years after the
            # focal year
            treatments_sample_long = treatments_sample_long[
                treatments_sample_long.year > focal_year
            ]

            # Get counts of pixels in different levels of severity that are not low
            # for each year
            treat_agg_sample = treatments_sample_long.groupby(
                ["year", column_treat], as_index=False
            ).grid_id.count()

            # Sum all units that are not low severity class and just build a
            # year and count dataframe.
            treat_means_df = (
                treat_agg_sample[treat_agg_sample[column_treat] >= high_class]
                .groupby("year", as_index=False)
                .grid_id.sum()
            )

            # Rename the column to something more meaningful
            treat_means_df.rename(columns={"grid_id": "treat_count"}, inplace=True)

            # Add number of treated pixels
            treat_means_df["treat_sample_size"] = len(treat_sample)

        else:
            raise ValueError(
                "Invalid outcome option. Available options are 'rr' or a pd.DataFrame with outcomes and a `grid_id` and `year` columns."
            )

        # Now, use the synth control weights per year and calculate the means for
        # each year outcomes. For the RR case we want the number of grid_ids in
        # other fire category that is not low severity. This is the denominator
        # of the RR estimator.
        try:
            # Subset weights to only the focal year. We will use this treatment
            # group to calculate the lagged effects in the rest of the years
            # (focal_year + 1 onwards).
            weights_year = weights[weights.focal_year == focal_year]
            if isinstance(outcomes, pd.DataFrame):
                # Filter outcomes to year of interest and to control units.
                # Notice we also leave the focal year in the control group as we
                # did the same in the treatment for this raw ATT calculation.
                outcomes_control = outcomes[
                    (outcomes.grid_id.isin(weights_year.grid_id.tolist()))
                    & (outcomes.year >= focal_year)
                ]

                # Merge to get aligned dataset with weights
                outcomes_control = outcomes_control.merge(weights_year, on="grid_id")

                # Group by year (all years larger or equal than the focal year
                # and calculate the weighted average for each year). This represents
                # the weighted average of the control group for each year after the
                # focal year treatment. For some reason, after apply the column
                # takes `None` as name (this is a pandas bug, I think).
                weighted_outcomes = outcomes_control.groupby(
                    "year", as_index=False
                ).apply(
                    lambda df: np.average(
                        df[outcome_var],
                        weights=df["weights"],
                    )
                )

                # Rename this to something more meaningful
                weighted_outcomes.rename(columns={None: "control_mean"}, inplace=True)

            elif outcomes == "rr":
                # To build the RR estimator denominator we do not need an outcome
                # Just like in the treatment, we want to track the control group
                # severity classes across time. Thus, we cannot do a simple count,
                # but a weighted one (this is the same as adding the weights as
                # is just a weighted sum of ones!)

                control_sample = weights_year.grid_id.tolist()

                # Reshape data to make the subsetting easier!
                treatments_sample = treatments[treatments.grid_id.isin(control_sample)]
                treatments_sample_long = pd.wide_to_long(
                    treatments_sample,
                    stubnames=column_names,
                    i="grid_id",
                    j="year",
                    sep="_",
                ).reset_index()

                # Remove past treatment years and focus on years after the focal
                # year
                treatments_sample_long = treatments_sample_long[
                    treatments_sample_long.year > focal_year
                ]

                # Merge with weights for that focal year
                treatments_sample_long = treatments_sample_long.merge(
                    weights_year, on="grid_id"
                )

                # Get weighted counts of the pixels in different levels of severity
                # that are not low for each year
                control_agg_sample = treatments_sample_long.groupby(
                    ["year", column_treat], as_index=False
                ).weights.sum()

                # Sum to all that is not low severity class and just build a year
                # and count dataframe.
                weighted_outcomes = (
                    control_agg_sample[control_agg_sample[column_treat] >= high_class]
                    .groupby("year", as_index=False)
                    .weights.sum()
                )

                # Rename the column to something more meaningful
                weighted_outcomes.rename(
                    columns={"weights": "control_count"}, inplace=True
                )

                # Add number of control pixels
                weighted_outcomes.loc[:, "control_sample_size"] = weights_year[
                    "weights"
                ].sum()

            else:
                raise ValueError(
                    "Invalid outcome option. Available options are 'rr' or a pd.DataFrame with outcomes and a `grid_id` and `year` columns."
                )
        except Exception as e:
            log.error(f"Error calculating estimator for focal year {focal_year}: {e}")

        # Merge treatment and control to calculate estimator
        # Notice that on this merge we will have a lag 0 or not depending on
        # the selection of the outcome. If the `rr` option is in, then we will
        # not have the focal_year in the sample of years, thus we will have a
        # lag starting in 1, rather than zero.
        estimator = treat_means_df.merge(weighted_outcomes, on="year")

        if not estimator.empty:
            estimator.loc[:, "focal_year"] = focal_year
            estimator["lag"] = estimator["year"] - focal_year

            if isinstance(outcomes, pd.DataFrame):
                estimator["att"] = estimator["treat_mean"] - estimator["control_mean"]
            else:
                # Calculate the RR estimator
                estimator["att"] = (
                    estimator["treat_count"] / estimator["treat_sample_size"]
                ) / (estimator["control_count"] / estimator["control_sample_size"])

            # Scale the estimator
            if scale:
                estimator.loc[:, "att"] = estimator["att"] * scale

        results.append(estimator)

    df = pd.concat(results)

    # Pool the estimates if passes
    if pooling:
        df = pooling_estimates(df, **kwargs)

    return df


def pooling_estimates(
    df: pd.DataFrame, cluster_var: str, formula: str, rr: bool = False
) -> pd.DataFrame:
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
    if rr:
        model_all_years = smf.glm(
            formula=formula,
            data=df,
            family=sm.families.NegativeBinomial(),
        )
    else:
        model_all_years = smf.ols(formula=formula, data=df)

    result_all_years = model_all_years.fit()
    coefs_all_years = result_all_years.predict(pd.DataFrame({"lag": lags}))

    # Model using OLS for each Jackknife sample and store in nd-array (samples, lags)
    arr = np.zeros((len(lists), lags.size))  # Fill array of sample / lags
    for jk_sample in lists:
        try:
            if rr:
                model = smf.glm(
                    formula=formula,
                    data=df[df[cluster_var].isin(jk_sample)],
                    family=sm.families.NegativeBinomial(),
                )
                result = model.fit()
            else:
                model = smf.ols(
                    formula=formula,
                    data=df[df[cluster_var].isin(jk_sample)],
                )
                result = model.fit()

            # Calculate predictions for each lag
            lag_df = pd.DataFrame({"lag": lags})
            coefs = result.predict(lag_df)

            arr[lists.index(jk_sample), :] = coefs.values

        except Exception as e:
            log.error(f"Error in Jackknife estimation: {e}")

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
