import os
import warnings

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyfixest as pf
import statsmodels.formula.api as smf
import xarray as xr
from joblib import Parallel, delayed
from prescribed.utils import expand_grid, grouper, prepare_template
from tqdm import tqdm


def make_model(
    linked_data: pd.DataFrame,
    formula: str,
    bootstrap: bool = False,
    k: int = 999,
    mask="sum_severity",
    predict=False,
    new_data=None,
):
    """Create a model using the linked data

    Use the linked data to create a model that can be used to estimate the
    effect of smoke on health outcomes.

    Parameters
    ----------
    linked_data : pd.DataFrame
        DataFrame with the linked data
    formula : str
        Formula to use in the model
    bootstrap : bool, optional
        Whether to perform bootstrapping, by default False
    k : int, optional
        Number of bootstrap samples, by default 999
    mask : str, optional
        Mask to filter coefficients, by default "sum_severity"

    Returns
    -------
    model: pyfixest.OLS or np.array
    """
    fes = None

    if len(formula.split("|")) > 1:
        fes = formula.split("|")[1].strip()

    # Function to perform bootstrapping for a single iteration
    def bootstrap_iteration(
        i,
        linked_data,
        formula,
        fes=None,
        mask=None,
        predict=False,
        new_data=None,
    ):
        sample = linked_data.sample(frac=1, replace=True)
        try:
            if fes is not None:
                model = pf.feols(
                    fml=formula,
                    data=sample,
                    fixef_tol=1e-3,
                    vcov={"CRV1": fes},
                )
            else:
                model = smf.ols(formula, data=sample).fit()

            if predict:
                # yeah, it says coefs, but they're predictions
                coefs_ = model.predict(newdata=new_data)
            else:
                # Extract the coefficients and transform to an array
                coefs_ = model.coef()
                coefs_ = coefs_[coefs_.index.str.contains(mask)].values

            return coefs_
        except Exception as e:
            print(e)
            return None

    # If predict, do bootstrap sample
    if bootstrap:
        # Run the bootstrapping process in parallel
        results = list(
            tqdm(
                Parallel(n_jobs=-1, return_as="generator")(
                    delayed(bootstrap_iteration)(
                        i, linked_data, formula, fes, mask, predict, new_data
                    )
                    for i in range(k)
                ),
                total=k,
                desc="Bootstrapping model results/coefficients",
            )
        )
        # Filter out None results and update coef_arr
        valid_results = [res for res in results if res is not None]

        if predict:
            coef_arr = np.array(valid_results)
        else:
            coef_arr = np.zeros((len(valid_results), 2))
            coef_arr[: len(valid_results), :] = np.array(valid_results)

        return coef_arr

    else:
        if fes is not None:
            model = pf.feols(
                fml=formula,
                data=linked_data,
                fixef_tol=1e-5,
                vcov={"CRV1": fes},
            )
        else:
            model = smf.ols(formula, data=linked_data).fit()

        return model


def make_predictions_bootstrap(
    severity_df, name_var, out_var, coefs, return_pandas=True
):
    """Create predictions using regression coefficients

    Use simulated severity array to create predictions for each year passed
    in the coefficients array.

    Parameters
    ----------
    severity : np.array
        Array of simulated severity values
    coefficients : pd.DataFrame
        DataFrame with the coefficients and standard errors of the model

    Returns
    -------
    predictions : np.array
        Array of predictions for each year
    """

    if not {"sim", "event_id", "year_treat", name_var}.issubset(
        severity_df.columns
    ):
        raise KeyError(
            f"Dataframe must have columns {name_var} and sim. Cannot run bootstrap"
        )

    # Pivot data to be wide (i -> events, j -> simulation_index)
    # Pivot dataframe to have sim as coluns and sum_dnbr as values
    severity_array = severity_df.pivot(
        index=["event_id", "year_treat"],
        columns="sim",
        values=name_var,
    )

    # Store index for later and transform to numpy array
    index = severity_array.index
    severity_array = severity_array.to_numpy()

    # To do: for now the quadratic transformation is not implemented and we're
    # just hardcoding stuff and make it work assuming we always have a quadratic
    # transformation. np.vander doesn't play well with > 1-D arrays.
    severity_squared = severity_array**2

    # Now this guy is (n_events, 2,  n_sims)
    severity_stacked = np.stack([severity_array, severity_squared], axis=1)

    # Do the dot product
    preds = np.einsum("ijk,jk->ik", severity_stacked, coefs.T)

    if return_pandas:
        preds = pd.DataFrame(
            preds, index=index, columns=severity_df.sim.unique()
        )

        # df.stack would be a faster way to achieve this, but the resulting
        # names are rather cryptic, so melt has more control over that. Haven't
        # test if this is good if df is massive.
        preds = pd.melt(
            preds.reset_index(),
            var_name="sim",
            value_name=out_var,
            id_vars=["event_id", "year_treat"],
        )

    return preds


def calculate_benefits(
    discount_rates,
    treat_severity,
    coefs,
    path=None,
    n_treats=None,
    average_treats=False,
    **kwargs,
):
    """ " Calculate benefits from a specific size policy under different discount rates

    This function will calcualte the total benefit of a prescribed policy using
    different parameters, including the size of the treatment, the relationship
    between the emissions and the severity and the discount rate.
    """

    # Get the simulation data merged with the observed dnbr. This dnbr data is
    # expected to be identical to the data we have used throughout the analysis
    # We calculate 3 tables:
    # 1. `dnbr_data` with the data severity data for each pixel, fire, event in
    #    the data
    # 2. `df` with all the simulations, the severity data and the severity
    #    benefits using the simulated data
    # 3. `sim_severity` with the severity data for each simulation and by year
    #    and event. This is the data we use to calculate the benefits of the
    #    policy

    if path is not None:
        path_db = os.path.join(path)
    else:
        path_db = f"../data/policy_no_spill_{n_treats}"

    dnbr_data = duckdb.query("""
    with dnbr_data as (
    select 
            grid_id, 
            year, 
            Event_ID as event_id, 
            Incid_Name as event_name, 
            dnbr 
    from '../data/dnbr.parquet' 
    where year > 2010
    )
    select event_id, 
        event_name,      
        year,
        sum(dnbr) as sum_dnbr
    from dnbr_data
    group by event_id, event_name, year
    """).to_df()

    df = duckdb.query(f"""
    WITH simulation_data AS (
        SELECT *
        FROM '{path_db}/*.parquet'
        where sim is not null
    ), 
    dnbr_data AS (
    select 
            grid_id, 
            year, 
            Event_ID as event_id, 
            Incid_Name as event_name, 
            dnbr 
    from '../data/dnbr.parquet' 
    where year > 2010
    ), 
    dnbr_event_agg as (
    select event_id, 
        event_name,      
        year,
        sum(dnbr) as sum_dnbr
    from dnbr_data
    group by event_id, event_name, year
    ), 
    benefits_grid_simulation as (
    SELECT  s.grid_id, 
            s.year_treat, 
            s.year, 
            d.event_id, 
            d.event_name, 
            s.sim,
            s.coeff, 
            d.dnbr,
            case when (d.dnbr + s.coeff) < 0 
                    then 0 
                    else d.dnbr + s.coeff 
            end as sim_benefit
    from simulation_data s 
    inner join dnbr_data d
    on s.grid_id = d.grid_id 
        and s.year = d.year
    ), 
    benefits_event_agg as (
    select event_id,
        year,
        year_treat, 
        event_name,
        sim,
        sum(sim_benefit) as sum_benefit_event
    from benefits_grid_simulation
    group by event_id, year, event_name, year_treat, sim
    ),
    total_event_benefits as (
    select d.event_id,
        d.event_name,
        d.year,
        d.sum_dnbr,
        coalesce(b.year_treat, 0) as year_treat,
        coalesce(b.sim, 0) as sim,
        coalesce(b.sum_benefit_event, 0) as sum_benefit_event,
        coalesce(d.sum_dnbr - b.sum_benefit_event, d.sum_dnbr) as simulated_sum_dnbr
    from benefits_event_agg b
    INNER JOIN dnbr_event_agg d
    on b.event_id = d.event_id and b.year = d.year)
    select * from total_event_benefits
    """).to_df()

    # Results from the query above cotain only the events where treatments and
    # fire happened. Thus, all MTBS events are not covered here. We want to have
    # all the events and have the counterfactual and the observed data together
    # with all the simulation runs. `df_sims` is a dataframe with all the combos
    df_sims = (
        df[["year_treat", "sim"]]
        .drop_duplicates()
        .sort_values(by=["year_treat", "sim"])
        .reset_index(drop=True)
    )

    dnbr_data_cross = dnbr_data.merge(df_sims, how="cross")

    # We need to make sure that we don't have any contamination. This should be
    # true by design, but the `year_treat` should be always less than the year.
    dnbr_data_cross = dnbr_data_cross[
        dnbr_data_cross.year > dnbr_data_cross.year_treat
    ]

    # Now bring all the data together!
    simulation_data = dnbr_data_cross.merge(
        df[
            [
                "event_id",
                "year",
                "year_treat",
                "sim",
                "simulated_sum_dnbr",
                "sum_benefit_event",
            ]
        ],
        on=["event_id", "year", "year_treat", "sim"],
        how="left",
    )

    # Fill nans with 0: If we don't have simulated dnbr, it means no treatment
    # so we should keep the same observed dnbr value. Now, if this is the case
    # it also means there's no benefits, thus they are zero.
    simulation_data = simulation_data.assign(
        simulated_sum_dnbr=lambda x: x.simulated_sum_dnbr.fillna(
            simulation_data.sum_dnbr
        ),
        sum_benefit_event=lambda x: x.sum_benefit_event.fillna(0),
    )

    # Translate severity benefits into emissions using the coefficients
    emissions = make_predictions_bootstrap(
        simulation_data,
        name_var="sum_dnbr",
        out_var="preds_pm",
        coefs=coefs,
        return_pandas=True,
    )

    simulated_emissions = make_predictions_bootstrap(
        simulation_data,
        name_var="simulated_sum_dnbr",
        out_var="preds_sim_pm",
        coefs=coefs,
        return_pandas=True,
    )

    # Merge both back to simulated data
    simulation_data = simulation_data.merge(
        emissions, on=["event_id", "year_treat", "sim"], how="left"
    )

    simulation_data = simulation_data.merge(
        simulated_emissions,
        on=["event_id", "year_treat", "sim"],
        how="left",
    )

    # Remove bad predictions than don't make sense in theory. This is another
    # check, and should always in theory be true, but we are adding it here
    # just in case.
    simulation_data = simulation_data[
        simulation_data.preds_sim_pm <= simulation_data.preds_pm
    ]

    # Create a warning if the number of rows is different
    if len(simulation_data) != len(dnbr_data_cross):
        warnings.warn(
            "There are some rows that were removed from the simulation data"
        )

    # Calculate benefits for each simulation
    simulation_data["benefit"] = (
        simulation_data["preds_pm"] - simulation_data["preds_sim_pm"]
    )

    # Calculate the total benefits for each year, sim (aggregate to the State
    # level)s
    benefits = simulation_data.groupby(
        ["year", "year_treat", "sim"], as_index=False
    )["benefit"].sum()

    # Now calculate the costs of policy emissions! These are the same across all
    # years and simulations, although they might different given uncertainty in
    # predictions (here that change with the uncertainty option in the
    # `make_predictions` function).
    total_years = benefits.year_treat.unique().shape[0]

    new_data = pd.DataFrame(
        {
            "sum_severity": np.repeat(
                np.array(np.array([treat_severity] * n_treats).sum()),
                total_years,
            ),
            "year": benefits.year_treat.unique(),
            "total_pixels": np.repeat(np.array([n_treats]), total_years),
            "total_days": np.repeat(np.array([8]), total_years),
        }
    )

    # Get predictions for each sim index and treatment year
    costs = make_model(
        new_data=new_data,
        **kwargs,
    )

    # Trasnform to a pandas dataframe and merge with the benefits
    costs_df = pd.DataFrame(costs.T)
    costs_df.index = new_data.year

    # Rename year to year_treat
    costs_df = costs_df.reset_index().rename(columns={"year": "year_treat"})

    costs_df = pd.melt(
        costs_df,
        var_name="sim",
        value_name="policy_cost",
        id_vars="year_treat",
    )

    benefits = benefits.merge(
        costs_df,
        on=["sim", "year_treat"],
        how="left",
    )

    # An alternative way to calculate PV benefits is to take the average for each
    # period of treatment and average on the lagged effect of the treatment.
    # This means that in lag 1 we have all the first years after the treatment
    # from year 2010 to 2020, and in lag 2 we have the second years after the
    # treatment from year 2010 to 2019, and so on. This is a more robust way to
    # calculate the benefits of the policy.
    if average_treats:
        benefits["lag"] = benefits.year - benefits.year_treat

        mean_benefits = benefits.groupby(["lag", "sim"], as_index=False)[
            ["benefit", "policy_cost"]
        ].mean()

        pv_list = []
        for discount_rate in discount_rates:
            for sim, sim_df in mean_benefits.groupby("sim"):
                for lag in sim_df.lag.unique():
                    # Create discount stream for the range of years in the data
                    years_policy = sim_df[sim_df.lag <= lag].shape[0]
                    discount_stream = 1 / (1 + discount_rate) ** np.arange(
                        1, years_policy + 1
                    )

                    # Calculate the PV benefits
                    pv = np.sum(
                        sim_df[sim_df.lag <= lag].benefit.values
                        @ discount_stream
                    )
                    ratio = pv / np.unique(sim_df.policy_cost)[0]

                    pv_list.append(
                        pd.DataFrame(
                            {
                                "lag": [lag],
                                "sim": [sim],
                                "pv": [pv],
                                "ratio": [ratio],
                                "discount_rate": [discount_rate],
                            }
                        )
                    )
        # Concatenate all the dataframes with PV/C ratios
        pv = pd.concat(pv_list)

    else:
        # Assume individual PV for each policy year.
        pv_list = []
        for discount_rate in discount_rates:
            for idx, df_year in benefits.groupby(["year_treat", "sim"]):
                year, sim = idx

                # Create discount stream for the range of years in the data
                years_policy = df_year.year.unique().shape[0]
                discount_stream = 1 / (1 + discount_rate) ** np.arange(
                    1, years_policy + 1
                )

                # Calculate the PV benefits
                pv = np.sum(df_year.benefit.values @ discount_stream)
                ratio = pv / np.unique(df_year.policy_cost)[0]

                pv_list.append(
                    pd.DataFrame(
                        {
                            "year_treat": [year],
                            "sim": [sim],
                            "pv": [pv],
                            "ratio": [ratio],
                            "discount_rate": [discount_rate],
                        }
                    )
                )

        # Concatenate all the dataframes with PV/C ratios
        pv = pd.concat(pv_list)
        pv["lag"] = np.abs(pv.year_treat - pv.year_treat.max()) + 1

    return pv, simulation_data


def simulation_data(
    template: str | pd.DataFrame,
    land_type: str | pd.DataFrame,
    roads: list | pd.DataFrame,
    only_roads: bool = True,
    buf: int = 4_000,
    road_type: str | None = "secondary",
    crs: str = "EPSG:3310",
) -> pd.DataFrame:
    """Set up data ready for Rx fire simulations

    This function will take a template of points and then subset them to get a
    sampling space that is close to roads by different land types.

    Parameters
    ----------
    template : geopandas.GeoDataFrame
        A geopandas dataframe with the template data
    land_type : pd.DataFrame
        A pandas dataframe with the land type data
    roads : geopandas.GeoDataFrame
        A geopandas dataframe with the road data
    buf : int
        The buffer distance to use for the roads

    Returns
    -------
        pd.DataFrame
            A pandas dataframe with the simulation data
    """

    cols_interest = [
        "lat",
        "lon",
        "grid_id",
        "land_type",
    ]

    # Load all datasets
    if isinstance(template, str):
        template = (
            prepare_template(template).groupby("grid_id").first().reset_index()
        )
        template = gpd.GeoDataFrame(
            template,
            geometry=gpd.points_from_xy(template.lon, template.lat),
            crs=crs,
        )

    if isinstance(land_type, str):
        land_type = pd.read_feather(land_type).drop(
            columns=["lat", "lon"], errors="ignore"
        )

    if only_roads:
        cols_interest = cols_interest + ["name", "fclass"]

        # Load the roads
        if isinstance(roads, list):
            roads = [gpd.read_file(road) for road in roads]
            roads = [road[road.fclass == road_type] for road in roads]
            roads = pd.concat(roads)

        # Buffer the roads
        roads["geometry"] = roads.to_crs(crs).buffer(buf)

        # Spatial join the template and the roads
        template = template.overlay(roads, how="intersection")

    # Merge with the land_type
    template = template.merge(land_type, on="grid_id")

    # Remove repeated pixels in the template
    # This can happens as windy roads or close roads will have the same pixel
    # in its buffer
    template = template.drop_duplicates(subset=["grid_id"])

    return template[cols_interest]


def sample_rx_years(
    template: str | pd.DataFrame,
    treat_data: pd.DataFrame,
    fire_data: pd.DataFrame,
    estimates: pd.DataFrame | str,
    size_treatment: int | None,
    spillovers: bool = False,
    spillover_size: int | None = 1_000,
    spillover_estimates: pd.DataFrame | None = None,
    start_year: int = 2010,
    sample_n: int = 100,
    crs: str = "EPSG:3310",
) -> pd.DataFrame:
    """Sample years for Rx fires

    This function will take the template data with buffered roads and sample
    points from the subset to select as treatments. The sample can also be of
    a specific size in which case we draw a square buffer in meters around the
    sampled point and get all the grid_ids that are within that buffer.

    Sampling is done without replacement, so if a pixels is selected in a year
    it won't be selected again for other years.

    Parameters
    ----------
    template : geopandas.GeoDataFrame or str
        A geopandas dataframe with the template data
    treat_data : pd.DataFrame
        A pandas dataframe with the treatment data
    fire_data : pd.DataFrame
        A pandas dataframe with the MTBS treatment data
    size_treatment : int
        The size of the treatment in meters
    sample_n : int
        The number of samples to take per year
    start_year : int
        The year to start sampling from. This would be the first year of
        treatment
    crs: str
        The coordinate reference system to use. Default is California Albers
        with meters as units.

    Returns
    -------
        pd.DataFrame
            A pandas dataframe with the sampled years
    """

    # Keep columns of interest at the end!
    cols = ["grid_id", "lat", "lon", "year", "year_treat", "coeff", "land_type"]

    # Load all datasets
    if isinstance(template, str):
        template = prepare_template(template, years=[2000, 2022])
        template = gpd.GeoDataFrame(
            template,
            geometry=gpd.points_from_xy(template.lon, template.lat),
            crs=crs,
        )
        template = template[template.year >= start_year]

    # Load fire data
    if isinstance(fire_data, str):
        fire_data = pd.read_feather(fire_data).drop(
            columns=["spatial_ref"], errors="ignore"
        )

    # Load ATT results from SC
    if isinstance(estimates, str):
        estimates = pd.read_csv(estimates)

    # Merge treats with the template so we can loop over years!
    template = template.merge(
        treat_data.drop(columns=["lat", "lon"], errors="ignore"), on="grid_id"
    )
    template = template.merge(fire_data, on=["lat", "lon", "year"], how="left")

    # Subset for the years we are interested in the treatment data
    template = template[template.year >= start_year]

    # Create weight for sampling. We want to downweight to sample units w/
    # fires in that year
    template = template[template.Event_ID.isin([np.nan, "nodata"])]

    # If spillovers, we need to buffer the points and get all the pixels that
    # are within that buffer
    if spillovers:
        if not isinstance(template, gpd.GeoDataFrame):
            template = gpd.GeoDataFrame(
                template,
                geometry=gpd.points_from_xy(template.lon, template.lat),
            )
            template.crs = crs

    # Sample the data by year without replacement across years
    sampled_years = []
    sample_grids = []
    for g_idx, df in tqdm(
        template.groupby("year"),
        desc="Sampling years",
        total=template.year.nunique(),
        leave=False,
    ):
        # Clean df and remove all the grids that were sampled before
        df = df[~df.grid_id.isin(sample_grids)]
        sample = df.sample(
            n=sample_n,
            axis=0,
            replace=False,
        )

        # If spillovers, add them to the sample using a buffer around each of
        # the sampled points
        if spillovers:
            spill = sample.copy()
            spill["geometry"] = spill.buffer(spillover_size, cap_style="square")
            spill = (
                template[template.year == g_idx]
                .drop(
                    columns=[
                        "lat",
                        "lon",
                        "year",
                        "Ig_Date",
                        "Event_ID",
                        "Incid_Type",
                        "Incid_Name",
                        "land_type",
                    ],
                    errors="ignore",
                )
                .overlay(spill.drop(columns=["grid_id"]), how="intersection")
            )
            spill = spill.drop_duplicates(subset=["grid_id"])

            # Remove the spillovers from future samples
            sample_grids.extend(spill.grid_id.unique())

            # Remove initial sample from the spill data
            spill = spill[~spill.grid_id.isin(sample.grid_id)]

            # Create sample grid to track spillovers (we are going to just focus
            # on the spillovers for now, and later add the initial treatment
            # sample later)
            spill_grid = expand_grid(
                {
                    "grid_id": spill.grid_id.unique(),
                    "year": range(g_idx + 1, 2023),
                }
            )

            # Add location and other stuff back
            spill_grid = spill_grid.merge(
                spill[["grid_id", "lat", "lon"]],
                on="grid_id",
                how="left",
            )
            # Bit of a hack to track the spillovers!
            spill_grid["land_type"] = 100

            # Add estimates!
            spillover_estimates_year = spillover_estimates.copy(deep=True)
            spillover_estimates_year["year"] = (
                g_idx + spillover_estimates_year.year
            )
            spill_grid = spill_grid.merge(
                spillover_estimates_year, on=["year"], how="left"
            )

            # Add the treatment year
            spill_grid["year_treat"] = g_idx

            # Randomily pick within the CIs -- assuming normality
            spill_grid["coeff"] = np.random.normal(
                loc=spill_grid.coef, scale=spill_grid.se
            )

        # Assert that old grid_id in sample_grids are not in the new sample
        # Notice here than if the spillovers are added, then we are only
        # checking for treatments, so is possible that this assertion is not
        # true when the spillovers are added. Now, a spillover pixel is not the
        # same as a treatment pixel, so we leave this assertion untouched.
        # assert len(set(sample_grids).intersection(set(df.grid_id))) == 0

        # Not the same grids on each year
        sample_grids.extend(sample.grid_id.unique())

        # Follow these sample in time! (notice is hardcoded to 2011 to 2022)
        sample_grid = expand_grid(
            {"grid_id": sample.grid_id.unique(), "year": range(g_idx + 1, 2023)}
        )

        # Get the land type back
        sample = sample_grid.merge(
            sample[["grid_id", "land_type", "lat", "lon"]],
            on="grid_id",
            how="left",
        )

        # Change lags to years! (we want to translate from the lags to the
        # years)
        estimates_year = estimates.copy(deep=True)
        estimates_year["year"] = g_idx + estimates_year.year

        # Add coeffs and the high and low CIs so we can sample values later!
        sample = sample.merge(
            estimates_year, on=["land_type", "year"], how="left"
        )
        sample["year_treat"] = g_idx

        # Randomily pick within the CIs
        sample["coeff"] = np.random.normal(loc=sample.coef, scale=sample.se)

        # Increase the size of treatment here. We are not developing this much
        # as usually these treatments are sub 1-square km. But we can increase
        # the size of the treatment to see how the spillovers behave [WIP]
        if size_treatment > 1000:
            sample["geometry"] = sample.buffer(
                size_treatment, cap_style="square"
            )
            sample = sample.overlay(df, how="intersects")
            sample = sample.drop_duplicates(subset=["grid_id"])

        if spillovers:
            sample = pd.concat([sample, spill_grid])

        sampled_years.append(sample[cols])

    sampled_years = pd.concat(sampled_years)

    # Make sampled years a dataframe
    sampled_years.drop(columns=["geometry"], errors="ignore", inplace=True)

    return sampled_years


def run_simulations(
    template: str | pd.DataFrame,
    sim_data: str | pd.DataFrame,
    results: str | pd.DataFrame,
    fire_data: str | pd.DataFrame,
    save_path: str,
    num_sims: int = 100,
    step_save: int = 10,
    dims: list = ["lat", "lon", "year", "year_treat"],
    format: str = "netcdf",
    **kwargs,
) -> None:
    """Execute experiment simulations!

    This function is just a loop-wrapper around the sample_rx_years function to
    run all the simulations several times and store the simulated every number
    of steps. The saving pattern can be set by changing the step_save parameter
    that will define the number of simulations per file and num_sims will define]
    the global number of simulations per experiment.

    Thus, if we run 100 simulations with a `step_save = 10`, then we will get
    `100/10 = 10` files with 10 simulations each.

    Parameters
    ----------
    template : str
        The path to the template data
    sim_data : str
        The path to the simulation data
    results : str
        The path to the results data
    fire_data : str
        The path to the fire data
    save_path : str
        The path to save the simulations
    num_sims : int
        The number of simulations to run
    step_save : int
        The number of simulations to save
    dims : list
        The dimensions to use for the simulations

    Returns
    -------
        None.
    """

    # Make sure save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Supress warnings
    import warnings

    warnings.filterwarnings("ignore")

    # Run all the simulations and save for each group
    for group_sims in tqdm(
        grouper(range(1, num_sims), step_save),
        total=int(num_sims / step_save),
        desc="Grouping simulations",
    ):
        arr_lst = []
        for idx in group_sims:
            test_arr = sample_rx_years(
                template=template,
                treat_data=sim_data,
                estimates=results,
                fire_data=fire_data,
                **kwargs,
            )

            if format == "netcdf":
                test_arr = test_arr.set_index(dims)[["coeff"]].to_xarray()

                # Add dimension to the array
                test_arr = test_arr.expand_dims({"sim": [idx]})
                arr_lst.append(test_arr)

            elif format == "parquet":
                test_arr["sim"] = idx
                arr_lst.append(test_arr)

        if format == "netcdf":
            # Concatenate all the arrays
            filename = os.path.join(
                save_path, f"sim_{group_sims[0]}_{group_sims[-1]}.parquet"
            )
            test_arr = xr.concat(arr_lst, dim="sim").to_netcdf(filename)

        elif format == "parquet":
            filename = os.path.join(
                save_path, f"sim_{group_sims[0]}_{group_sims[-1]}.parquet"
            )
            test_arr = pd.concat(arr_lst)

            # Select only the columns we care about
            test_arr = test_arr[dims + ["grid_id", "land_type", "coeff", "sim"]]

            test_arr.to_parquet(filename, index=False)
        else:
            ValueError(
                f"Format {format} not supported. Only netcdf and parquet"
            )

    return None
