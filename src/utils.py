import numpy as np
import pandas as pd


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
