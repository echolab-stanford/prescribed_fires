import hashlib
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from tqdm import tqdm

import joblib
import contextlib


def expand_grid(dict_vars):
    """Create cartesian product of a set of vectors and return a datafarme

        This function calculates the cartesian product of a set of vectors and
        return a tabular data structure, just as R's expand.grid function.
    a
        Parameters
        ----------
        dict_vars : dict
            Dictionary containing the vectors to be combined. The keys are the
            column names and the values are the vectors.

        Returns
        -------
        pandas.DataFrame
            Dataframe containing the cartesian product of the vectors
    """
    mesh = np.meshgrid(*dict_vars.values())
    data_dict = {var: m.flatten() for var, m in zip(dict_vars.keys(), mesh)}

    return pd.DataFrame(data_dict)


def transform_array_to_xarray(data, transform, extra_dims=None):
    """Transform numpy array to xarray using an affine transform (from rasterio)

    This function takes a numpy array with a transform affine matrix and return
    an xarray with latitude and longitude coordinates and extra dimensions if
    passed.

    TODO:
    ----
    CRS is not passed as an option, but that would be a nice addition!

    Parameters
    ----------
    data : np.array
        Array with data
    transform : affine.Affine
        Affine transform matrix
    extra_dims : list, optional
        List with extra dimensions to add to the xarray

    Returns
    -------
        An xarray with the data and coordinates
    """

    height = data.shape[0]
    width = data.shape[1]

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    # Concat into arrays
    lons = np.array(xs)
    lats = np.array(ys)

    # Create xarray
    coords = {"lat": lats[:, 0], "lon": lons[0, :]}

    if extra_dims:
        coords.update(extra_dims)

        # Take care of array dimensions. Dimensions should be at minimum 2D
        # by construction. If extra dims are passed, we need to add a new axis
        # to the data array if the data is not there.
        # For example:
        # data.shape = (10, 10) # lon and lat
        # data.shape = (10, 10, 1) # lon, lat and time.
        # We need to pass an array with the last shape if we have extra dims.

        extra_dims_len = len(extra_dims)

        if len(data.shape) == 2:
            data = data[(...,) + (np.newaxis,) * extra_dims_len]

    return xr.DataArray(data, coords=coords, dims=list(coords.keys()))


def prepare_template(template_path, years=[2000, 2023]):
    """Aux function to prepare the template as a cartesian product of time and space

    This function expects a raster template with 1 for land and 0 for water as
    the one generated by the `prepare_template` function in the `src/data` module.
    The function will create a cartesian product of time and space to create a
    template with all the pixels in the sample.

    Args:
        template_path (str): Path to the template raster
        years (list): List with the first and last year of the sample: [start, end]

    Returns:
        pd.DataFrame
    """

    # Load template to start merging
    if isinstance(template_path, str):
        template = rioxarray.open_rasterio(template_path)
    else:
        template = template_path

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

    # Create grid for all years in the sample: cartesian product of time and
    # observations
    template_expanded = expand_grid(
        {"grid_id": template_df.grid_id, "year": np.arange(years[0], years[1] + 1)}
    )

    # Add lat and lon to the expanded grid
    template_expanded = template_expanded.merge(
        template_df[["grid_id", "lat", "lon"]], on="grid_id"
    )

    return template_expanded


def generate_run_id(parameters):
    """Generate a unique run id from the parameters"""
    params_string = str(parameters)

    return hashlib.md5(params_string.encode()).hexdigest()


def calculate_fire_pop_dens(
    geoms: gpd.GeoDataFrame,
    pop_raster_path: Union[dict, str],
    buffer: int = 10000,
    date_col: str = "Ig_Date",
    mask: Optional[Union[gpd.GeoDataFrame, str]] = None,
    template: Optional[xr.core.dataarray.DataArray] = None,
):
    """Calculate population density within a buffer of each geometry

    This function will calculate the population density column for a particular
    dataframe of geometries. It will use population rasters (such as CIESIN) to
    calculate the population density within a user-defined buffer for each
    geometry. As some population products change each 5-10 years, this function
    will use a raster that is closest to the year of the geometry using a year
    column and a dictionary of raster paths. If the dictionary is not passed, it
    will be built using CIESIN population rasters naming defaults.

    The function will return a data frame with the population density for each
    buffer:
     - Event_ID: The ID of the geometry
     - Ig_Date: The date of the geometry
     - Incid_Name: The name of the geometry
     - Incid_Type: The type of the geometry
     - buffer: The buffer distance in meters
     - total_pop: The total population within the buffer
     - mean_pop: The mean population density within the buffer
     - donut: The geometry of the buffer without the geometry

    Parameters
    ----------
    geoms : gpd.GeoDataFrame
        A GeoDataFrame with geometries to calculate population density
    pop_raster_path : Union[dict, str]
        A dictionary of raster paths for population data or a single path
    buffer : int, optional
        The buffer distance in meters, by default 10000 (10 km)
    date_col : str, optional
        The column in the GeoDataFrame with the year of the geometry, by default "Ig_Date"
    mask : Optional[rioxarray.raster_array.RasterArray], optional
        A raster to mask the population raster, by default None
    """

    # If a dictionary is not passed, build it from CIESIN population rasters
    if isinstance(pop_raster_path, str):
        pop_raster_path = {
            int(p.stem.split("_")[-3]): rioxarray.open_rasterio(p)
            for p in Path(pop_raster_path).glob("*.tif")
        }

    if mask is not None:
        if isinstance(mask, str):
            mask = gpd.read_file(mask).to_crs(4326)
        else:
            # Reproject mask to the population raster CRS (by default is 4326)
            # maybe adding a pop_raster_crs parameter in the future.
            mask = mask.to_crs(4326)

        for key, val in pop_raster_path.items():
            ds_attrs = val.attrs

            ds_mask = val.rio.clip(mask.geometry.values, drop=True, invert=False)

            if template is not None:
                # Reproject the mask to the template
                ds_mask = ds_mask.rio.reproject_match(template)
            else:
                ds_mask = ds_mask.rio.reproject(3310)

            # Mask the population raster with np.nan for population outside the mask
            ds_mask = xr.where(ds_mask == ds_attrs["_FillValue"], np.nan, ds_mask)
            ds_mask = ds_mask.rio.write_crs(3310)

            pop_raster_path[key] = ds_mask
    else:
        print("Without a mask the process can take longer than expected")

    # Get the year of the geometry
    geoms[date_col] = pd.to_datetime(geoms[date_col])
    geoms["year"] = geoms[date_col].dt.year

    # Project to meters in the CA projection Albers
    geoms = geoms.to_crs("EPSG:3310")
    geoms["buffer"] = geoms.buffer(buffer)

    # Get the population raster closest to the year of the geometry
    arr_years = np.apply_along_axis(
        lambda x: min(pop_raster_path.keys(), key=lambda y: np.abs(y - x)),
        1,
        geoms["year"].values.reshape(-1, 1),
    )
    geoms["pop_raster_year"] = arr_years

    pop_dens_geom = []
    for _, row in tqdm(
        geoms.iterrows(), total=geoms.shape[0], desc="Calculating population density..."
    ):
        # Open the raster and crop to the geometry
        ds = pop_raster_path[row["pop_raster_year"]].squeeze()

        # Clip the raster to the geometry and build a donut
        clip_geom = ds.rio.clip([row["buffer"]], drop=True, invert=False)
        donut = clip_geom.rio.clip([row["geometry"]], drop=True, invert=True)

        # Do the same with geometry
        donut_geom = row["buffer"].difference(row["geometry"])

        # Calculate the pop of the geometry
        total_pop = donut.sum().values
        mean_pop = donut.mean().values

        # Transform to df
        pop_dens_geom.append(
            gpd.GeoDataFrame(
                {
                    "Event_ID": row["Event_ID"],
                    "Ig_Date": row["Ig_Date"],
                    "Incid_Name": row["Incid_Name"],
                    "Incid_Type": row["Incid_Type"],
                    "buffer": buffer,
                    "total_pop": total_pop,
                    "mean_pop": mean_pop,
                    "donut": donut_geom,
                    "geometry": row["geometry"],
                },
                index=[0],
            )
        )

    # Return a DataFrame with the population density
    df = pd.concat(pop_dens_geom)
    df.crs = "EPSG:3310"

    return df


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument

    Taken from: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
