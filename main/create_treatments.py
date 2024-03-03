""" Use MTBS data to create a treatment identification based on a template

This function takes a MTBS shapefile and rasterizes it using a user defined
template. The function will spit out a NetCDF file with the rasterized
polygons using the "Event_ID" columns as the values of the data. To do this,
we only use the numeric part of the event ID. Thus:

Event: CA3607412018819840329 will be converted to 3607412018819840329

Each year will be a dimension in the NetCDF file.
"""

import argparse
import os
from functools import partial

import geopandas as gpd
import pandas as pd
import rioxarray
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_image


def create_treatments(mtbs_shapefile, template, save_path):
    """Create rasterized treatments from MTBS dataset using a grid template

    This function takes a MTBS shapefile and rasterizes it using a user defined
    template. The function will spit out a Feather tabular file with the rasterized
    polygons using the "Event_ID" columns as the values of the data. Additionally,
    the file will contain a selection of the columns from the original MTBS:
        - Ig_Date
        - Incid_Name
        - Incid_Type

    Each year will be a dimension in the NetCDF file.

    Sources:
    -https://www.mtbs.gov/direct-download

    Parameters
    ----------
    mtbs_shapefile : str
        Path to the MTBS shapefile to process
    template : str
        Path to the template file to reproject to
    save_path : str
        Path to save the resulting output

    Returns
    -------
    None
        Saves a Feather file to the save_path
    """

    # Read in data frame and filter for California
    mtbs = gpd.read_file(mtbs_shapefile)
    mtbs["Ig_Date"] = pd.to_datetime(mtbs["Ig_Date"])

    # Open template as xarray
    template = rioxarray.open_rasterio(template)

    # Filter for wildfires and prescribed fires in California
    wildfires = mtbs[
        (mtbs.Event_ID.str.contains("CA"))
        & (mtbs.Incid_Type.isin(["Wildfire", "Prescribed Fire"]))
    ]

    # Reproject to same proj as template (CA Albers)
    wildfires = wildfires.to_crs(4326)
    wildfires_meters = wildfires.to_crs(template.rio.crs.to_epsg())

    # Create year column to add time dimension
    wildfires_meters["year"] = wildfires_meters.Ig_Date.dt.year

    # Create a enums dict with category values
    enums = {"Event_ID": wildfires_meters["Event_ID"].unique().tolist()}

    # Use geocube to rasterize the shapefile
    wildfire_arr = make_geocube(
        vector_data=wildfires_meters,
        measurements=["Event_ID"],
        like=template,
        categorical_enums=enums,
        group_by="year",
        rasterize_function=partial(rasterize_image, all_touched=True),
    )

    # Transform indices into categories within the array
    event_id_str = wildfire_arr["Event_ID_categories"][
        wildfire_arr["Event_ID"].astype(int)
    ].drop_vars("Event_ID_categories")
    wildfire_arr["Event_ID"] = event_id_str

    # Rename xarray dimensions: x -> lon, y -> lat
    wildfire_arr = wildfire_arr.rename({"x": "lon", "y": "lat"})

    # Convert xarray to dataframe and add correct event_id data back!
    wildfire_df = (
        wildfire_arr.drop_dims(["Event_ID_categories"]).to_dataframe().reset_index()
    )

    # Merge with subset data from original MTBS
    wildfire_merge = wildfires_meters[
        ["Event_ID", "Ig_Date", "Incid_Name", "Incid_Type"]
    ].merge(wildfire_df, on="Event_ID", how="right")

    # Save the file
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    wildfire_merge.to_feather(os.path.join(save_path, "treatments_mtbs.feather"))

    return None


if __name__ == "__main__":
    # Define the parser
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mtbs",
        type=str,
        help="Path to the MTBS shapefile to process",
        required=True,
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to the template file to reproject to",
        required=True,
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the resulting output",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the function
    create_treatments(
        mtbs_shapefile=args.mtbs,
        template=args.template,
        save_path=args.save_path,
    )

    print(f"File saved to {args.save_path}!")
