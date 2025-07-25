import numpy as np
import pandas as pd
from .report import report_treatments

from ..utils import prepare_template


def classify_dnbr(
    df: pd.DataFrame, dnbr: str = "dnbr", class_name: str = "class_dnbr"
) -> pd.DataFrame:
    """
    Find the DNBR class from the earliest fire per each grid using the
    DNBR threshold values from Key & Benson (2006)

    Parameters:
    ----------
        df (pd.DataFrame): DataFrame with the DNBR values
        dnbr (str): Column name with the DNBR values
        class_name (str): Column name for the DNBR class

    Returns:
    -------
        pd.DataFrame: DataFrame with the DNBR class
    """
    conditions = [
        (df[dnbr] < 100),
        (df[dnbr] >= 100) & (df[dnbr] < 270),
        (df[dnbr] >= 270) & (df[dnbr] < 440),
        (df[dnbr] >= 440) & (df[dnbr] < 660),
        (df[dnbr] >= 660),
    ]
    # zero-index to match the low-intensity level to be 1.
    choices = [0, 1, 2, 3, 4]

    # Asigning FRP class to each fire
    df.loc[:, class_name] = np.select(conditions, choices, default=np.nan)

    return df


def classify_frp(
    df: pd.DataFrame, frp: str = "frp", class_name: str = "class_frp"
) -> pd.DataFrame:
    """Find the FRP class from the earliest fire per each grid using the
    FRP threshold values (Ichoku et al. 2014)

    Parameters:
    ----------
        df (pd.DataFrame): DataFrame with the FRP values
        frp (str): Column name with the FRP values
        class_name (str): Column name for the FRP class

    Returns:
    -------
        pd.DataFrame: DataFrame with the FRP class
    """
    conditions = [
        (df[frp] > 0) & (df[frp] < 100),
        (df[frp] >= 100) & (df[frp] < 500),
        (df[frp] >= 500) & (df[frp] < 1000),
        (df[frp] >= 1000) & (df[frp] < 1500),
        df[frp] >= 1500,
    ]
    choices = [1, 2, 3, 4, 5]

    # Asigning FRP class to each fire
    df.loc[:, class_name] = np.select(conditions, choices, default=np.nan)

    return df


def fill_treatment_template_frp(
    frp: str, template_path: str, treatments_path: str
):
    """Aux function to merge template with FRP data and build intensity based treatments

    Just like `fill_treatment_template_mtbs`, but based for FRP. This is using
    the FRP processed data from `prescribed.extract.process_modis_frp` and will
    simply select all pixels with FRP > 0 as treated. We will use the MTBS to
    get Event_IDs, but we will call whatever is not in the MTBS as a "frp_fire_event"
    and keep in the treatment dataset.

    Parameters:
    ----------
        frp (str): Path to the feather file with FRP data
        template_path (str): Path to the template raster
        treatments_path (str): Path to the feather file with treatments

    Returns:
    -------
        pd.DataFrame: DataFrame with the template filled with treatments and
        additional columns:
            - grid_id: Unique identifier for each pixel
            - year: Year of the observation
            - frp: max FRP value for each pixel-year
            - treat: 1 if the pixel is treated, 0 otherwise
            - Event_ID: Event ID from the MTBS dataset. If the pixel is not in the
                MTBS dataset, it will be called "frp_fire_event".
            - lat: Latitude of the pixel
            - lon: Longitude of the pixel
    """

    # Load template
    template_expanded = prepare_template(template_path)

    # Load treatments and FRP data
    treatments = pd.read_feather(treatments_path).drop(
        columns=["spatial_ref"], errors="ignore"
    )

    frp = pd.read_feather(frp).drop(columns=["spatial_ref"], errors="ignore")

    # Remove all no-events in treatments
    treatments = treatments[treatments["Event_ID"] != "nodata"]

    # Merge with template to clean treatments (they're full of water!)
    treatments = template_expanded.merge(treatments, on=["lat", "lon", "year"])

    # Merge with MTBS treats and keep all the FRP readings
    frp = frp.merge(treatments, on=["grid_id", "year"], how="left")
    frp.loc[(frp.frp > 0) & (frp.Event_ID.isna()), "Event_ID"] = (
        "frp_fire_event"
    )

    # Create treatments columns (here is just fire!)
    frp["treat"] = np.where(frp.frp > 0, 1, 0)

    return frp


def fill_treatment_template_mtbs(
    template_path: str,
    treatments_path: str,
    staggered: bool = False,
    verbose: bool = False,
    min_count_treatments: int = 2,
    query: str = None,
    **kwargs,
) -> pd.DataFrame:
    """Aux function to merge template with MTBS-like treatment data in feather format

    This function takes a template raster and a feather file with treatments and
    merge them together to design the treatment allocations defined in the paper.
    The template can use a pandas query if we need to remove specific pixels from
    the MTBS template (i.e. no prescribed fires).

    In the paper, we rely on treatments be staggered meaning that if we see a
    pixel more than once (our sample!), we will count the first period as the
    initial period of treatment and the subsequent periods as evaluation. Notice
    this definition depends on the dataset we use:
        - MTBS data will define these treatments as the overlap of fire polygons
        - FRP will use pixel-level fire detections to define treatments.

    If the `staggered` option is used then the function will calculate the relative
    year variable (year - min_treat_year) to keep track of the years since the first
    treatment, and will remove all observations will more than three observations
    in time (i.e. we only want twice exposed pixels).

    Parameters:
    ----------
        template_path (str): Path to the template raster
        treatments_path (str): Path to the feather file with treatments
        query (str): A pandas query to filter the treatments,
            e.g. "Event_ID != 'Prescribed Fire'"
        staggered (bool): If True, the function will calculate the relative year
            variable (year - min_treat_year) to keep track of the years since the
            first treatment, and will remove all observations will more than three
            observations in time (i.e. we only want twice exposed pixels).
        verbose (bool): If True print reports on the data
        min_count_treatments (int): Minimum number of treatments to keep a pixel
            in the sample. Default is 2.


    Returns:
    -------
        pd.DataFrame: DataFrame with the template filled with treatments and
        additional columns:
            - grid_id: Unique identifier for each pixel
            - year: Year of the observation
            - lat: Latitude of the pixel
            - lon: Longitude of the pixel
            - min_treat_year: First year of treatment for each pixel
            - rel_year: Relative year since the first treatment
    """

    template_expanded = prepare_template(template_path, **kwargs)

    # Load treatments and MTBS data
    treatments = pd.read_feather(treatments_path).drop(
        columns=["spatial_ref"], errors="ignore"
    )

    if query:
        treatments = treatments.query(query)

    # Merge with template to clean treatments (they're full of water!)
    treatments = template_expanded.merge(
        treatments, on=["lat", "lon", "year"], how="left"
    )

    # Create treatments columns (without query, everything in the treatment is
    # going to be 1 by default)
    treatments = treatments.assign(
        treat=np.select(
            [treatments["Event_ID"] == "nodata", treatments["Event_ID"].isna()],
            [0, 0],
            default=1,
        ),
    )

    treatments["count_treats"] = treatments.groupby("grid_id").treat.transform(
        "cumsum"
    )

    if staggered:
        # Reduce sample! Only keep the pixels that have only one that treatments
        treatments = treatments[treatments.count_treats <= min_count_treatments]

        # Get the first year of treatment for each grid_id in the dataframe
        min_years = (
            treatments[treatments.treat == 1]
            .groupby("grid_id", as_index=False)
            .year.min()
            .astype(int)
            .rename({"year": "min_treat_year"}, axis=1)
        )

        # Calculate relative year variable (year - min_year)
        treatments = treatments.merge(min_years, on="grid_id", how="left")
        treatments["rel_year"] = (
            treatments.year - treatments["min_treat_year"]
        ).astype(float)

    if verbose:
        report_treatments(treatments)

    return treatments


def build_lhs(covariates_dict: dict[str, dict]) -> pd.DataFrame:
    """Aux function to build the LHS of the design matrix

    Our regressions need DVs! This function uses the template to build the DV and
    later merge with the treatments. Ideally, we would like to use treatments here
    to reduce the size of the dataset as we can subset only the pixels we need, but
    estimation needs to be flexible too, so this can be a big dataset!

    Parameters:
    ----------
        covariates_dict (dict): Path to covariates feather files. The code expects
        a dict with covariate keys. These keys should be "dnbr" or "frp", otherwise
        an error will be raised. Since classification is not always desired (imagine
        using data that is already classified), you can pass an additional parameter
        in the dictionary: "classify" that can be a boolean. If True, the function
        will classify the dataset using default thresholds hardcoded in this function.

        An example of the dictionary is:
        {
            "dnbr": {"path": "data/dnbr.feather", "classify": False},
            "frp": {"path": data/frp.feather", "classify": True},
        }

    Returns:
    -------
        pd.DataFrame: DataFrame with the template filled with the covariates and
        additional columns:
            - grid_id: Unique identifier for each pixel
            - year: Year of the observation
            - lat: Latitude of the pixel
            - lon: Longitude of the pixel
    """

    datasets = ["dnbr", "frp"]

    if not set(covariates_dict.keys()).issubset(datasets):
        raise KeyError(
            f"Keys in covariates_path should be in {datasets}, got {covariates_dict.keys()}"
        )

    # Load covariates
    data_dict = {}
    for key in covariates_dict.keys():
        data_dict[key] = pd.read_feather(covariates_dict[key]["path"]).drop(
            columns=["spatial_ref"], errors="ignore"
        )

    # Categorize covariates
    for key, data in data_dict.items():
        if key == "frp":
            if "year" not in data.columns:
                data["year"] = data.time.dt.year

            frp_groupped = data.groupby(
                ["grid_id", "year"], as_index=False
            ).frp.max()
            if covariates_dict[key]["classify"]:
                # Find the FRP class from the earliest fire per each grid using the
                # FRP threshold values (Ichoku et al. 2014):
                conditions = [
                    (frp_groupped.frp > 0) & (frp_groupped.frp < 100),
                    (frp_groupped.frp >= 100) & (frp_groupped.frp < 500),
                    (frp_groupped.frp >= 500) & (frp_groupped.frp < 1000),
                    (frp_groupped.frp >= 1000) & (frp_groupped.frp < 1500),
                    frp_groupped.frp >= 1500,
                ]
                choices = [1, 2, 3, 4, 5]

                # Asigning FRP class to each fire
                frp_groupped["class_frp"] = np.select(
                    conditions, choices, default=np.nan
                )
                data_dict[key] = frp_groupped

        elif key == "dnbr":
            # Change column type to facilitate merge with frp later
            data.loc[:, "grid_id"] = data["grid_id"].astype(int)

            # Drop weird points (see analysis/weird_points.ipynb)
            data = data[~data.grid_id.isna()]

            if covariates_dict[key]["classify"]:
                # Find the DNBR class from the earliest fire per each grid using the
                # DNBR threshold values from Key & Benson (2006):
                conditions = [
                    (data.dnbr < 100),
                    (data.dnbr >= 100) & (data.dnbr < 270),
                    (data.dnbr >= 270) & (data.dnbr < 440),
                    (data.dnbr >= 440) & (data.dnbr < 660),
                    data.dnbr >= 660,
                ]
                # zero-index to match the low-intensity level to be 1.
                choices = [0, 1, 2, 3, 4]

                # Asigning FRP class to each fire
                data.loc[:, "class_dnbr"] = np.select(
                    conditions, choices, default=np.nan
                )
            else:
                data.rename(columns={"dnbr": "class_dnbr"}, inplace=True)

            data_dict[key] = data

        else:
            raise KeyError(f"Key {key} not found in datasets")

    return data_dict


def treatment_schedule(
    treatment_template: pd.DataFrame,
    treatment_fire: dict,
    no_class: bool = False,
) -> pd.DataFrame:
    """Create treatment dataframe organized by year and grid_id

    This function combines treatment data from an MTBS-like dataset and combines
    it with severity and intensity data to create a treatment schedule based on
    severity and intensity classes rather than in fire ocurrance. This function
    will also pivot the data to return a wide-format dataset that can be merged
    with wide-like covariates and used in synthtetic control balancing.

    Importantly, the treatments are going to be organized along columns to have
    a direct definition of focal years, which are the years of treatment assignment
    in the original dataset under conditions defined by the user.

    Parameters:
    ----------
    treatment_template (pd.DataFrame): DataFrame with the template filled with
        treatments and additional columns:
            - grid_id: Unique identifier for each pixel
            - year: Year of the observation
            - lat: Latitude of the pixel
            - lon: Longitude of the pixel
            - min_treat_year: First year of treatment for each pixel
            - rel_year: Relative year since the first treatment
    treatment_fire (dict): A dictionary with different treatment allocations based
    on severity and intensity classes.
    no_class (bool): If True, the function will not use the class columns to

    Returns:
    -------
    pd.DataFrame: DataFrame with the treatment schedule organized by year and
    grid_id in wide format.
    """

    # Merge treatments and fire data
    for key, value in treatment_fire.items():
        if key == "dnbr":
            treatment_template = treatment_template.merge(
                value,
                right_on=["grid_id", "event_id"],
                left_on=["grid_id", "Event_ID"],
                how="left",
            )
        elif key == "frp":
            treatment_template = treatment_template.merge(
                value,
                on=["grid_id", "year"],
                how="left",
            )
        else:
            raise KeyError(f"Key {key} not found in datasets")

    # Set all NaN values as zero for the class columns:
    # class_frp and class_dnbr
    treatment_template["class_frp"] = treatment_template["class_frp"].fillna(0)
    treatment_template["class_dnbr"] = treatment_template["class_dnbr"].fillna(
        0
    )

    # If the treatment definition is not given by classes, we can just shut these
    # and leave them as 1, such as when multiplied by the treatment they're always
    # 1.
    if no_class:
        treatment_template["class_frp"] = 1
        treatment_template["class_dnbr"] = 1

    # Drop coordinate columns
    treatment_template.drop(
        columns=[
            c for c in treatment_template.columns if "lon" in c or "lat" in c
        ],
        inplace=True,
    )

    # Pivot data to wide format
    treats_wide = pd.pivot(
        treatment_template,
        index="grid_id",
        columns="year",
        values=["treat", "class_dnbr", "class_frp"],
    )
    treats_wide.columns = [f"{col}_{year}" for col, year in treats_wide.columns]
    treats_wide = treats_wide.reset_index()

    return treats_wide
