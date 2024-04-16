"""Run balancing for a focal year and a specific land type

This script is a wrapper for the run_balancing function. It prepares the data and runs the balancing for a specific land type.

Land-types are defined as a integer:
    1: "Agricultural",
    2: "Conifer",
    3: "Conifer-Hardwood",
    4: "Developed",
    5: "Exotic Herbaceous",
    6: "Exotic,Tree-Shrub",
    7: "Grassland",
    8: "Hardwood",
    9: "No Data",
    10: "Non-vegetated",
    11: "Riparian",
    12: "Shrubland",
    13: "Sparsely Vegetated"
"""

import argparse
from functools import reduce

import numpy as np
import pandas as pd
import rioxarray
from sklearn.preprocessing import MinMaxScaler

from src.estimate.run_balancing import run_balancing
from src.utils import expand_grid


def prepare_data(path_covars, path_template, path_treatments):
    # Load template to start merging
    template = rioxarray.open_rasterio(path_template)

    # Notice template is 1 for land and 0 for water
    template_df = (
        template.rename({"x": "lon", "y": "lat"})
        .to_dataframe(name="id_template")
        .reset_index()
        .dropna()
    )
    template_df["grid_id"] = template_df.index

    # Remove the water pixels
    template_df = template_df[template_df.id_template == 1]

    # Create grid for all years in the sample
    template_expanded = expand_grid(
        {"grid_id": template_df.grid_id, "year": np.arange(2000, 2023)}
    )

    # Add lat and lon to the expanded grid
    template_expanded = template_expanded.merge(
        template_df[["grid_id", "lat", "lon"]], on="grid_id"
    )

    # Load treatments and MTBS data
    treatments = pd.read_feather(path_treatments)

    if "spatial_ref" in treatments.columns:
        treatments.drop(columns=["spatial_ref"], inplace=True)

    # Merge with template to clean treatments (they're full of water!)
    treatments = template_expanded.merge(
        treatments, on=["lat", "lon", "year"], how="left"
    )

    # Create treatments columns (both wildifre and prescribed are 1!)
    treatments = treatments.assign(
        treat=np.select(
            [treatments["Event_ID"] == "nodata", treatments["Event_ID"].isna()],
            [0, 0],
            default=1,
        ),
    )

    ### Merge with output data: intensity and severity with their low types
    #### 1. Intensity with the FRP tresholds (Ichoku et al., 2014)

    # Merge with FRP data
    # Just take wildfires, because we don't have prescribed fire severity data
    # We already remove them before, but we are being extra explicit here
    treatments = treatments[treatments.Incid_Type != "Prescribed Fire"]

    # Load intensity FRP data
    frp = pd.read_feather(f"{path_covars}/frp_concat.feather")
    frp["year"] = frp.time.dt.year
    frp_groupped = frp.groupby(["lat", "lon", "year"], as_index=False).frp.max()

    # Find the FRP class from the earliest fire per each grid using the
    # FRP threshold values (Ichoku et al. 2014):
    conditions = [
        frp_groupped.frp < 100,
        (frp_groupped.frp >= 100) & (frp_groupped.frp < 500),
        (frp_groupped.frp >= 500) & (frp_groupped.frp < 1000),
        (frp_groupped.frp >= 1000) & (frp_groupped.frp < 1500),
        frp_groupped.frp >= 1500,
    ]
    choices = [1, 2, 3, 4, 5]

    # Asigning FRP class to each fire
    frp_groupped["class_frp"] = np.select(conditions, choices, default=np.nan)

    # Merge with wildfires data
    treatments = treatments.merge(frp_groupped, on=["lat", "lon", "year"], how="left")
    treatments["class_frp"] = treatments["class_frp"].fillna(0)

    # #### 2. Severity from Landsat and scaling from -1 to 1

    # Load intensity FRP data
    dnbr = pd.read_feather(f"{path_covars}/dnbr_alt.feather").drop(columns=["band"])

    # Scale DNBR values
    dnbr_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(
        dnbr.dnbr.values.reshape(-1, 1)
    )
    dnbr["dnbr_scaled"] = dnbr_scaled

    # Find the DNBR class from the earliest fire per each grid using the
    # DNBR threshold values:
    conditions = [
        (dnbr.dnbr >= 0) & (dnbr.dnbr < 0.1),
        (dnbr.dnbr >= 0.1) & (dnbr.dnbr < 0.270),
        (dnbr.dnbr >= 0.270) & (dnbr.dnbr < 0.440),
        (dnbr.dnbr >= 0.440) & (dnbr.dnbr < 0.660),
        dnbr.dnbr >= 0.660,
    ]
    choices = [1, 2, 3, 4, 5]

    # Asigning FRP class to each fire
    dnbr["class_dnbr"] = np.select(conditions, choices, default=np.nan)

    # Merge with wildfires data
    treatments = treatments.merge(dnbr, on=["lat", "lon", "Event_ID"], how="left")
    treatments["class_dnbr"].fillna(0, inplace=True)

    treats_wide = pd.pivot(
        treatments,
        index="grid_id",
        columns="year",
        values=["treat", "class_dnbr", "class_frp"],
    )
    treats_wide.columns = [f"{col}_{year}" for col, year in treats_wide.columns]
    treats_wide = treats_wide.reset_index()

    dict_paths = {
        "prism": f"{path_covars}/prism.feather",
        "disturbances": f"{path_covars}/disturbances.feather",
        "dem": f"{path_covars}/dem.feather",
        "frp": f"{path_covars}/frp_wide.feather",
        "land_type": f"{path_covars}/land_type.feather",
    }

    # Load datasets and merge with template
    data = []
    for key, path in dict_paths.items():
        print(key)
        df = pd.read_feather(path)

        # Remove columns if present
        if "lat" in df.columns:
            df = df.drop(columns=["lat", "lon"])
        data.append(df)

    # Merge all datasets
    df = reduce(lambda x, y: x.merge(y, on="grid_id", how="left"), data)
    df = df.merge(treats_wide, on="grid_id")

    # Save some memory
    del data

    return df


if __name__ == "__main__":
    # Add command-line options from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_covars", type=str, required=True)
    parser.add_argument("--path_template", type=str, required=True)
    parser.add_argument("--path_treatments", type=str, required=True)
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--focal_year", type=int, default=2019)
    parser.add_argument("--land_type", type=int, default=2)
    args = parser.parse_args()

    # Prepare data
    df = prepare_data(
        path_covars=args.path_covars,
        path_template=args.path_template,
        path_treatments=args.path_treatments,
    )

    # Run balancing
    run_balancing(
        df=df[df.land_type.isin([args.land_type])].dropna(),
        focal_year=args.focal_year,
        treat_col=f"treat_{args.focal_year}",
        class_col=f"class_frp_{args.focal_year}",
        row_id="grid_id",
        reg_list=[0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0, 1],
        lr_list=[1, 0.1, 1e-2, 1e-3, 1e-4, 1e-5],
        intercept=True,
        niter=10_000,
        metrics=["smd", "asmd"],
        save_path=args.save_path,
    )