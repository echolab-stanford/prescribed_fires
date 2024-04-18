import sqlite3


def create_db_tables(db_path: str) -> None:
    """Create storage data tables for model balancing results"""

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create table for model results
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
        weights REAL,
        reg REAL,
        lr REAL,
        row_id INTEGER,
        model_run_id TEXT,
        focal_year INTEGER,
        land_type TEXT,
        PRIMARY KEY (model_run_id, focal_year, row_id, land_type)
        ON CONFLICT IGNORE
        )
        """
    )

    # Create table for covariate differences
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS std_diffs (
        std_unweighted_smd REAL,
        std_weighted_smd REAL,
        std_unweighted_asmd REAL,
        std_weighted_asmd REAL,
        reg REAL,
        lr REAL,
        model_run_id TEXT,
        focal_year INTEGER,
        covar TEXT,
        land_type TEXT,
        PRIMARY KEY (model_run_id, focal_year, covar, land_type)
        ON CONFLICT IGNORE
        )
        """
    )

    # Create table for loss
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS loss (
        loss REAL,
        lr_decay REAL,
        iter INTEGER,
        niter INTEGER,
        reg REAL,
        lr REAL,
        model_run_id TEXT,
        focal_year INTEGER,
        land_type TEXT,
        PRIMARY KEY (model_run_id, focal_year, land_type, niter)
        ON CONFLICT IGNORE
        )
        """
    )

    conn.close()

    return None
