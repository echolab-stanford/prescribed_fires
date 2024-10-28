import os
import pdb

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from prescribed.utils import expand_grid, grouper, prepare_template
from tqdm import tqdm


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

    # Subset for the years we are interested in the treatment data
    fire_data = fire_data[fire_data.year >= start_year]

    # Merge treats with the template so we can loop over years!
    template = template.merge(
        treat_data.drop(columns=["lat", "lon"], errors="ignore"), on="grid_id"
    )
    template = template.merge(fire_data, on=["lat", "lon", "year"], how="left")

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
            spill_grid["land_type"] = "spillover"

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
        assert len(set(sample_grids).intersection(set(df.grid_id))) == 0

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

        sampled_years.append(sample)

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
            sample_rx = sample_rx_years(
                template=template,
                treat_data=sim_data,
                estimates=results,
                fire_data=fire_data,
                **kwargs,
            )

            test_arr = sample_rx.copy()
            test_arr = test_arr.merge(
                template, on=["lat", "lon", "year", "grid_id"], how="right"
            )
            test_arr["coeff"] = test_arr["coeff"].fillna(0)

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

            test_arr.to_parquet(filename)
        else:
            ValueError(
                f"Format {format} not supported. Only netcdf and parquet"
            )

    return None
