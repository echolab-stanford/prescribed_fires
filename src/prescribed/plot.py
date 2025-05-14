import os
from functools import reduce
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from geopandas.tools import sjoin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def annotate_axes(ax, text, fontsize=18):
    """Placeholder for layout desing in matplotlib

    Taken from: https://matplotlib.org/stable/users/explain/axes/arranging_axes.html
    """
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="darkgrey",
    )


def plot_std_diffs(
    std_diffs_df,
    palette="Blues",
    drop_vars=None,
    ax=None,
    save_path=None,
    labels_x=True,
    labels_y=True,
    draw_cbar=True,
    label=None,
    rotation_x=45,
    sorted=True,
    vmax=None,
    vmin=None,
):
    """Plot Standarized Absolute Mean Differences as an array

    Parameters
    ----------
    std_diffs_df : pd.DataFrame or str
        DataFrame with the Standarized Absolute Mean Differences
    palette : str
        Color palette to use in the plot
    ax : matplotlib.axes.Axes, optional
        Axes object
    save_path : str, optional
        Path to save the plot, by default None. If None, the plot is not saved
    drop_vars : list, optional
        List of variables to drop from the plot, by default None
    labels_x : bool, optional
        Add labels to the x-axis, by default True
    labels_y : bool, optional
        Add labels to the y-axis, by default True
    draw_cbar : bool, optional
        Draw the colorbar, by default True
    label : str, optional
        Add a label to the plot in the top-left, by default None. If passed, you should pass a letter only.
    rotation_x : int, optional
        Rotation of the x-axis labels, by default 45

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """

    DEFAULT_ORDER = [
        "aspect",
        "curvature",
        "elevation",
        "slope",
        "ppt",
        "disturbances_no_disturbance",
        "vpdmin",
        "vpdmax",
        "frac_cover_tree",
        "frac_cover_shrub",
        "tdmean",
        "tmin",
        "tmax",
        "tmean",
        "frac_cover_bare",
        "count_fires",
        "frac_cover_herbaceous",
        "disturbances_unattributed_browning",
        "disturbances_fire",
        "disturbances_unattributed_greening",
        "disturbances_timber_harvest",
        "cummax_frp",
        "disturbances_drought_forest_die",
        "Intercept",
        "land_type",
    ]

    # Open file if a dataframe is not passed
    if isinstance(std_diffs_df, str):
        std_diffs_df = pd.read_csv(std_diffs_df)

    # Create a covar family for some of the variables that are timeseries
    # Add family_covar column to the dataframe to aggregate covariates
    std_diffs_df["family_covar"] = [
        "_".join([i for i in c.split("_") if not i.isdigit()])
        for c in std_diffs_df.covar.tolist()
    ]

    std_diffs_grouped = (
        std_diffs_df.groupby(["family_covar", "focal_year"], as_index=False)[
            [
                "std_weighted_asmd",
                "std_unweighted_asmd",
                "std_unweighted_smd",
                "std_weighted_smd",
            ]
        ]
        .mean()
        .sort_values(["std_unweighted_asmd"])
    )

    if drop_vars is not None:
        std_diffs_grouped = std_diffs_grouped[
            ~std_diffs_grouped.family_covar.isin(drop_vars)
        ]

        # Remove drop vars from the default order
        DEFAULT_ORDER = [var for var in DEFAULT_ORDER if var not in drop_vars]

    # Create pivot table to represent data as an array
    # Notice here we pick the standarized absolute mean differences! This is
    # because we want to see the variables that are more problematic in terms of
    # balance in an absolute scale, different than a Love plot.
    asmd_unweighted_array = pd.pivot_table(
        std_diffs_grouped,
        index="family_covar",
        columns="focal_year",
        values="std_weighted_asmd",
    )

    # Create a figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Adjust the space between the subplots
    plt.subplots_adjust(wspace=0, hspace=0.2)

    # Plot the array
    # Pre-sort the rows of the array to have the largest values at the bottom
    if sorted:
        sorted_arr = asmd_unweighted_array.loc[
            asmd_unweighted_array.mean(axis=1)
            .sort_values(ascending=False)
            .index
        ]
    else:
        sorted_arr = asmd_unweighted_array.loc[DEFAULT_ORDER]

    # Normalize the pallete to take a range from 0 to 1
    norm = plt.Normalize(
        vmin=np.min(sorted_arr) if vmin is None else vmin,
        vmax=np.max(sorted_arr) if vmax is None else vmax,
    )

    arr = sorted_arr.values
    cax = ax.imshow(arr, cmap=palette, norm=norm, aspect="auto")

    # Add a colorbar to the plot and a title to the colorbar
    if draw_cbar:
        cbar = plt.colorbar(cax)
        cbar.set_label("Absolute Standarized Mean Differences", size=12)

    # Add the labels to the x and y axis using the family_covar and focal_year from the std_diffs_grouped dataframe. The x labels should be in the bottom of the plot
    if labels_x:
        ax.set_xticks(
            np.arange(len(sorted_arr.columns)), labels=sorted_arr.columns
        )
    else:
        ax.set_xticks([])

    if labels_y:
        ax.set_yticks(range(len(sorted_arr.index)), labels=sorted_arr.index)
    else:
        ax.set_yticks([])

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation_x,
        ha="right",
        rotation_mode="anchor",
    )

    # Remove the box (frame) around the plot
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add label to plot (left-top corner)
    if label is not None:
        ax.text(
            -1.5,
            1.10,
            label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="right",
        )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    return ax, arr.min(), arr.max()


def plot_loss_check(path_to_losses, best_model_path):
    """Plot losses

    This is not a plot meant for production, just a quick way to checck the CV
    losses of the models
    """

    paths = list(Path(path_to_losses).rglob("*.parquet"))

    best_model = pd.read_csv(best_model_path)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for path in paths:
        loss = pd.read_parquet(path)
        if loss.model_run_id.unique()[0] in best_model["model_run_id"].tolist():
            reg = loss.reg.unique()[0]
            lr = loss.lr

            # Plot loss
            ax[0].plot(loss.loss, label=f"Reg: {reg} - LR: {lr}")
            ax[1].plot(loss.lr_decay, label=f"Reg: {reg} - LR: {lr}")

            ax[0].set_ylabel("Loss")
            ax[0].set_xlabel("Iterations")
            ax[0].set_yscale("log")

            ax[1].set_ylabel("Learning Rate")
            ax[1].set_xlabel("Iterations")
            ax[1].set_yscale("log")

    # Make some space between the axis
    fig.tight_layout()

    # Apply the template
    for ax in ax.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))

    return ax


def plot_std_diffs_focal_year(
    std_diffs, focal_year, ax=None, save_path=None, drop_vars=None
):
    """Plot Standarized Mean Differences for a specific focal year"""

    # Create covar family if is not there
    if "family_covar" not in std_diffs.columns:
        # Create a covar family for some of the variables that are timeseries
        # Add family_covar column to the dataframe to aggregate covariates
        std_diffs["family_covar"] = [
            "_".join([i for i in c.split("_") if not i.isdigit()])
            for c in std_diffs.covar.tolist()
        ]

    std_diffs_year = std_diffs[std_diffs.focal_year == focal_year]

    # Group by family_covar and focal_year
    std_diffs_year = (
        std_diffs_year.groupby(["family_covar", "focal_year"], as_index=False)[
            [
                "std_weighted_asmd",
                "std_unweighted_asmd",
                "std_unweighted_smd",
                "std_weighted_smd",
            ]
        ]
        .mean()
        .sort_values(["std_unweighted_asmd"])
    )

    if drop_vars is not None:
        std_diffs_year = std_diffs_year[
            ~std_diffs_year.family_covar.isin(drop_vars)
        ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        std_diffs_year["std_weighted_smd"].sort_values(),
        std_diffs_year["family_covar"],
        "o",
        label="Weighted",
    )
    ax.plot(
        std_diffs_year["std_unweighted_smd"].sort_values(),
        std_diffs_year["family_covar"],
        "o",
        label="Unweighted",
    )
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(-0.1, color="black", linestyle="--")
    ax.axvline(0.1, color="black", linestyle="--")

    # Add more ticks to the x axis from -0.5 to 0.5
    ax.set_xticks(np.round(np.arange(-0.5, 0.5, 0.1), 1))
    ax.set_xticklabels(np.round(np.arange(-0.5, 0.5, 0.1), 1))

    # Add labels to axis
    ax.set_ylabel("Variables")
    ax.set_xlabel("Standardized Differences")

    # Add legend to the plot
    ax.legend()

    # Make plot nicer by removing top and right spines and moving the ticks outward
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines.left.set_position(("outward", 10))
    ax.spines.bottom.set_position(("outward", 10))
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Save the plot
    if save_path:
        plt.savefig(
            os.path.join(save_path, f"std_diffs_{focal_year}.png"),
            bbox_inches="tight",
            dpi=300,
        )


def plot_outcomes(
    dfs: list[pd.DataFrame],
    axes_names: list[str],
    cmaps: list[str],
    colors: list[str],
    labels: list[str],
    var_interest: str = "att",
    order: int = 1,
    dotted: int = 0,
    robust: bool = False,
    lowess: bool = False,
    legend: bool = False,
    ax: plt.Axes = None,
    label_fig: str = None,
    pooled: bool = False,
    reg_plot: bool = False,
    size: str | None = None,
    **kwargs,
) -> plt.Axes:
    """Plot outcomes for main results"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    if pooled:
        time_var = "year"
        for df, cmap, color, label in zip(dfs, cmaps, colors, labels):
            # Sort dfs to make sure plots are nice
            df_type = df.sort_values(time_var)

            df_type.plot(
                x=time_var,
                y="coef",
                color=color,
                legend=False,
                label=label,
                ax=ax,
            )
            ax.fill_between(
                df_type[time_var],
                df_type["low_ci"],
                df_type["high_ci"],
                color=color,
                alpha=0.2,
            )
    else:
        time_var = "lag"
        for df, cmap, color, label in zip(dfs, cmaps, colors, labels):
            if "focal_year" not in df.columns:
                raise KeyError(
                    "focal_year is not in the dataframe. Maybe you want to pass pooled=True"
                )

            # sns.scatterplot(
            #     x=time_var,
            #     y=var_interest,
            #     data=df[(df[time_var] >= 0) & (df["focal_year"] <= 2021)],
            #     hue="focal_year",
            #     palette=cmap,
            #     size="control_count",
            #     ax=ax,
            #     alpha=0.5,
            #     legend=True,
            # )

            df[(df[time_var] >= 0) & (df["focal_year"] <= 2021)].plot.scatter(
                x=time_var,
                y=var_interest,
                c="focal_year",
                cmap=cmap,
                ax=ax,
                alpha=0.5,
                s=size if size else None,
                colorbar=False,
            )
            if reg_plot:
                sns.regplot(
                    x=time_var,
                    y=var_interest,
                    data=df[(df["focal_year"] <= 2021)],
                    scatter=False,
                    ax=ax,
                    color=color,
                    label=label,
                    order=order,
                    lowess=lowess,
                    robust=robust,
                )

    if legend:
        ax.legend(loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    ax.axhline(dotted, color="black", linestyle="--", c="gray")

    # Set the x-axis ticks so they start at 1 and end at the end of the data

    # Get max value from the data
    max_val = max([df[time_var].max() for df in dfs])
    ax.set_xticks(range(1, np.int8(max_val) + 1))

    ax = template_plots(
        ax,
        ylab=axes_names[1],
        xlab=axes_names[0],
        label=label_fig if label_fig is not None else None,
        rotation_x=0,
        **kwargs,
    )

    return ax


def get_best_poly_fit(
    X,
    y,
    degrees,
    poly_col_index="sum_severity",
    min_rmse=float("inf"),
    min_deg=1,
):
    """
    Get the best polynomial fit for a given data.

    Parameters
    ----------
    X : np.array
        Features array
    y : np.array
        Target array
    poly_col_index : int
        Index of the column in X to apply the polynomial transformation
    degrees : list
        Degrees to test the polynomial fit
    min_rmse : float
        Minimum RMSE threshold
    min_deg : int
        Minimum degree threshold

    Returns
    -------
    dict
        Dictionary with the degrees, RMSEs and best degree
    """
    # Shuffle and split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    rmses = []
    for deg in degrees:
        # Train features
        poly_features = PolynomialFeatures(degree=deg, include_bias=True)

        # Transform only the specified column
        x_poly_train_col = poly_features.fit_transform(
            x_train[poly_col_index].values.reshape(-1, 1)
        )

        # Concatenate the transformed column with the rest of the features
        x_poly_train = np.concatenate(
            [
                x_poly_train_col,
                x_train,
            ],
            axis=1,
        )

        # Linear regression
        poly_reg = LinearRegression()
        poly_reg.fit(x_poly_train, y_train)

        # Compare with test data
        x_poly_test_col = poly_features.fit_transform(
            x_test[poly_col_index].values.reshape(-1, 1)
        )
        x_poly_test = np.concatenate(
            [
                x_poly_test_col,
                x_test,
            ],
            axis=1,
        )
        poly_predict = poly_reg.predict(x_poly_test)
        poly_mse = mean_squared_error(y_test, poly_predict)
        poly_rmse = np.sqrt(poly_mse)
        rmses.append(poly_rmse)

        # Cross-validation of degree
        if min_rmse > poly_rmse:
            min_rmse = poly_rmse
            min_deg = deg

    return {
        "degrees": degrees,
        "rmses": rmses,
        "best_degree": min_deg,
    }


def run_fit_curve(
    data, x_col, y_col, cmap, by=None, ax=None, plot=True, **kwargs
):
    """Run the polynomial fit for a given data, x and y columns

    Parameters
    ----------
    data : pd.DataFrame
        Data to fit
    x_col : str
        Column name to use as X
    y_col : str
        Column name to use as y
    by : str
        Column to subset fitting
    cmap : str
        Color map to use
    ax : matplotlib.Axes
        Axis to plot. Optional

    Returns
    -------
    dict
        Dictionary with the degrees, RMSEs and best degree
    """
    # Set seed for reproducibility
    np.random.seed(42)

    X = data[x_col].values.reshape(-1, 1)
    y = data[y_col].values

    if plot:
        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if by is not None:
        results_list = []
        for lt, color in zip(data[by].unique(), cmap):
            data_lt = data[data[by] == lt].dropna()
            X = data_lt[x_col].values.reshape(-1, 1)
            y = data_lt[y_col].values

            results = get_best_poly_fit(X, y, **kwargs)
            results_list.append(results)
            results[by] = lt

            # Plot RMSE
            if plot:
                ax.plot(
                    results["degrees"],
                    results["rmses"],
                    label=f"{lt} [$P_n$ = {results['best_degree']}]",
                    color=color,
                )
    else:
        results = get_best_poly_fit(X, y, **kwargs)

        # Plot RMSE
        if plot:
            ax.plot(
                results["degrees"],
                results["rmses"],
                label=f"[$P_n$ = {results['best_degree']}]",
            )

    # Remove axis
    if plot:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_position(("outward", 10))
        ax.spines["bottom"].set_position(("outward", 10))

        # Change axis names
        ax.set_yscale("log")
        ax.legend()
        ax.set_xlabel("Polynomial degree")
        ax.set_ylabel("RMSE")

    results = pd.DataFrame(results_list)[[by, "best_degree", "pred"]]

    if plot:
        out = (ax, results)
    else:
        out = results

    return out


def data_fire_plot(
    wide_treats: pd.DataFrame,
    frp: pd.DataFrame,
    dnbr: pd.DataFrame,
    emissions: pd.DataFrame | None = None,
    year: int | None = None,
    event_id: str | None = None,
    mtbs: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """Util function to build a dataframe ready for plotting combining data
    sources

    This function will read the treatments and DVs (FRP and dNBR) to build a
    dataframe ready for plotting. This includes several steps:
      - Merge data sources before adding year and grid_id if is not there
      - Filter by year and return complete dataframe as a gpd.GeoDataFrame ready
        for plotting. If no year is passed, it will return the complete
        dataframe without treatments by year.

    Parameters
    ----------
    wide_treats : pd.DataFrame
        Treatments data
    frp : pd.DataFrame
        FRP data
    dnbr : pd.DataFrame
        dNBR data
    year : int
        Year to filter. If None, it will return the complete dataframe without
        treatments by year
    event_id : str
        Event ID to filter. If None, it will return the complete dataframe
        without treatments by year
    mtbs : gpd.GeoDataFrame
        MTBS data

    Returns
    -------
    gpd.GeoDataFrame
        Data ready for plotting
    """
    id_cols = ["treat", "class_frp", "class_dnbr"]

    # Open data sources
    if isinstance(wide_treats, str):
        wide_treats = pd.read_feather(wide_treats)

    if isinstance(frp, str):
        frp = pd.read_feather(frp)

    if isinstance(dnbr, str):
        dnbr = pd.read_feather(dnbr)

        try:
            if isinstance(mtbs, str):
                mtbs = gpd.read_file(mtbs)
        except ValueError:
            raise ValueError(
                "You need to pass a path to the MTBS data if you pass a path to the dNBR data to assign year from the Event ID"
            )

    if isinstance(emissions, str) and emissions is not None:
        emissions = pd.read_feather(emissions)

    if year is not None:
        id_cols_year = [
            f"{col}_{y}" for col, y in zip(id_cols, [year] * len(id_cols))
        ]

        test_treats = wide_treats.reset_index()[id_cols_year + ["grid_id"]]
        test_treats["year"] = year

        # Merge with dnbr and frp
        frp_year = frp[frp.year == year]
        dnbr_year = dnbr[dnbr.year == year].drop(columns=["lat", "lon"])

        if emissions is not None:
            emissions_year = emissions[emissions.year == year].drop(
                columns=["lat", "lon"]
            )

        fires = frp_year.merge(dnbr_year, on=["grid_id", "year"])

        # Merge with the treatments
        fires = fires.merge(test_treats, on=["grid_id", "year"])

        # Merge with the emissions
        if emissions is not None:
            fires = fires.merge(
                emissions_year, on=["grid_id", "year"], how="left"
            )

        fires["treat_dnbr"] = (
            fires[f"treat_{year}"] * fires[f"class_dnbr_{year}"]
        )
        fires["treat_frp"] = fires[f"treat_{year}"] * fires[f"class_frp_{year}"]
        fires["treat_severity"] = np.where(fires["treat_dnbr"] == 1, 1, 0)
        fires["treat_intensity"] = np.where(fires["treat_frp"] == 1, 1, 0)

    else:
        # Reshape wide-treatments to long format to merge with the other data sources
        long_treatments = pd.wide_to_long(
            wide_treats,
            stubnames=id_cols,
            i="grid_id",
            j="year",
            sep="_",
        ).reset_index()

        reduc_list = (
            [frp, dnbr, emissions, long_treatments]
            if emissions is not None
            else [frp, dnbr, long_treatments]
        )

        fire = reduce(
            lambda x, y: pd.merge(x, y, on=["grid_id", "year"], how="left"),
            reduc_list,
        )

        fire["treat_severity"] = np.where(
            (fire["treat"] * fire["class_dnbr"]) == 1, 1, 0
        )
        fire["treat_intensity"] = np.where(
            (fire["treat"] * fire["class_frp"]) == 1, 1, 0
        )

    if event_id:
        # Transform to geopandas
        fire = fires[fires["Event_ID"] == event_id]
        fire = gpd.GeoDataFrame(
            fire, geometry=gpd.points_from_xy(fire.lon, fire.lat), crs=3310
        )
        fire = sjoin(fire, mtbs[mtbs["Event_ID"] == event_id])

    return fire


def template_plots(
    ax,
    ylab,
    xlab,
    diag=False,
    no_axis=False,
    log_y=False,
    label=None,
    label_pos=(-0.1, 1.2),
    axis_text=10,
    label_axis=12,
    vert=None,
    title=None,
    label_vert=None,
    label_vert_pos=None,
    rotation_x=0,
):
    """Axis template for matplotlib plots.

    This function takes a matplotlib axis and modifies it to make it look nicer
    and also to make plotting more succint than passing all these parameters
    every time we make a plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to modify
    ylab : str
        Label for the y-axis
    xlab : str
        Label for the x-axis
    diag : bool, optional
        Add a 45-degree line to the plot, by default False
    no_axis : bool, optional
        Remove all axis (good for maps), by default False
    log_y : bool, optional
        Add log-scale in y-axis, by default False
    label : str, optional
        Add label to plot (left-top corner), by default None. If passed, you
        should pass a letter only.
    axis_text : int, optional
        Size of the text in the axis, by default 10
    title : str, optional
        Title of the plot, by default None. Added to the top of the axis.
    vert : float, optional
        Add a vertical line to the plot, by default None
    label_vert : str, optional
        Label for the vertical line, by default None
    label_vert_pos : tuple, optional
        Position for the label of the vertical line, by default None. This is
        using the data axis, so you should pass the x and y position for the
        label in data units.
    rotation_x : int, optional
        Rotation of the x-axis labels, by default 90

    Returns
    -------
    matplotlib.axes.Axes
        Modified axis
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    # Remove all axis (good for maps)
    if no_axis:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    # Add a 45-degree line to the plot
    if diag:
        # Add a 45 deg line that takes the min and max from the plot
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="gray",
            linestyle="--",
        )

    # Add label to plot (left-top corner)
    if label is not None:
        label_x, label_y = label_pos
        ax.text(
            label_x,
            label_y,
            label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top",
            ha="right",
        )

    # Add log-scale in y-axis
    if log_y:
        ax.set_yscale("log")

    # Add labels
    ax.set_ylabel(ylab, fontsize=label_axis)
    ax.set_xlabel(xlab, fontsize=label_axis)

    # Change text in ticks
    ax.tick_params(labelsize=axis_text)

    if title is not None:
        ax.set_title(title, fontsize=axis_text)

    # Add vertical line
    if vert:
        ax.axvline(vert, color="gray", linestyle="--")

        if label_vert is not None:
            x, y = label_vert_pos
            ax.text(
                x,
                y,
                label_vert,
                fontsize=axis_text,
                color="gray",
                ha="right",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="center",
            )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotation_x,
        ha="center",
        rotation_mode="anchor",
    )

    return ax
