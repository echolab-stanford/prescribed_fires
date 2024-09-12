import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from prescribed.utils import prepare_template
from prescribed.utils import expand_grid


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

    # Load RR results from SC
    if isinstance(estimates, str):
        estimates = pd.read_csv(estimates)
    else:
        estimates = estimates.copy(deep=True)

    # Change lags to years!
    estimates["year"] = start_year + estimates.year

    # Subset for the years we are interested in the treatment data
    fire_data = fire_data[fire_data.year >= start_year]

    # Merge treats with the template so we get full years
    template = template.merge(
        treat_data.drop(columns=["lat", "lon"], errors="ignore"), on="grid_id"
    )
    template = template.merge(fire_data, on=["lat", "lon", "year"], how="left")

    # Create weight for sampling. We want to downweight to sample units w/
    # fires in that year
    template["weight"] = 100
    template.loc[
        (~template.Event_ID.isna()) & (template.year == 2010), "weight"
    ] = 0.1

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
        sample = df.sample(n=sample_n, weights="weight")

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

        # Add coeffs and the high and low CIs so we can sample values later!
        sample = sample.merge(estimates, on=["land_type", "year"], how="left")
        sample["year_treat"] = g_idx

        # Randomily pick within the CIs
        # sample["coeff"] = np.random.uniform(sample.low_ci, sample.high_ci)
        sample["coeff"] = np.random.normal(loc=sample.coef, scale=sample.se)

        if size_treatment > 1000:
            sample["geometry"] = sample.buffer(
                size_treatment, cap_style="square"
            )
            sample = sample.overlay(df, how="intersects")
            sample = sample.drop_duplicates(subset=["grid_id"])

        sampled_years.append(sample)

    sampled_years = pd.concat(sampled_years)

    return sampled_years
