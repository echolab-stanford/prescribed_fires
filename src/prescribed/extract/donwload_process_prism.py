"""
Download and process PRISM raster data
"""

import logging
import os
from datetime import timedelta
from ftplib import FTP

import numpy as np
import tqdm


def datetime_range(start, end, delta):
    current = start
    if not isinstance(delta, timedelta):
        delta = timedelta(**delta)
    while current < end:
        yield current
        current += delta


def download_prism_data_year(
    var_of_interest,
    path,
    date,
    all_year=True,
    year=None,
):
    """
    Download .bil data from the PRISM FTP server. This data is monthly and the
    function will retrieve all the monthly for the given year. The code can be
    simply modified to download a single day using the FTP /daily/, rather than
    /monthly/.

    Parameters
    ----------
    var_of_interest : str
        Variable of interest. Options are: tmin, tmax, tdmean, vpdmin, vpdmax,
        ppt, tmean
    path : str
        Path to save the data to
    year : int
        Year to download the data for
    date : datetime
        Date to download the data for
    all_year : bool
        If True, download all the months for the given year. If False, download
        a single day

    Returns
    -------
    str
        Path to the downloaded file
    """
    logger = logging.getLogger(__name__)

    if all_year:
        save_path = os.path.join(path, var_of_interest, str(year))

        if os.path.exists(save_path):
            logger.info("Directory exists!")
        else:
            os.makedirs(save_path)

        with FTP("prism.nacse.org") as ftp:
            ftp.login()
            ftp.cwd(f"/monthly/{var_of_interest}/{year}")

            filenames = ftp.nlst()
            for day_file in filenames:
                with open(os.path.join(save_path, day_file), "wb") as file:
                    ftp.retrbinary(f"RETR {day_file}", file.write)
    else:
        year = str(date.year)
        save_path = os.path.join(path, var_of_interest, year)
        string_date = date.strftime("%Y%m%d")
        path = f"PRISM_{var_of_interest}_stable_4kmD1_{string_date}_bil.zip"

        if os.path.exists(save_path):
            logger.info("Directory exists! Skip!")

        else:
            os.makedirs(save_path)

        with FTP("prism.nacse.org") as ftp:
            ftp.login()
            ftp.cwd(f"/monthly/{var_of_interest}/{year}")

            with open(os.path.join(save_path, path), "wb") as file:
                ftp.retrbinary(f"RETR {path}", file.write)

    return str(os.path.join(save_path, path))


if __name__ == "__main__":
    save_dir = "/mnt/sherlock/oak/prescribed_data/raw/prism"
    years = np.arange(1988, 2023, 1)

    for var in ["tmin", "tmax", "tdmean", "vpdmin", "vpdmax", "ppt", "tmean"]:
        for year in tqdm.tqdm(years, total=len(years), desc=var):
            download_prism_data_year(var, save_dir, year)
