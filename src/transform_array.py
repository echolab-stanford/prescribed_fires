import numpy as np
import rasterio
import xarray as xr


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
