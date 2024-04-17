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
        focal_year INTEGER
        PRIMARY KEY (model_run_id, focal_year, row_id)
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
        covar TEXT
        PRIMARY KEY (model_run_id, focal_year, covar)
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
        reg REAL,
        lr REAL,
        model_run_id TEXT,
        focal_year INTEGER
        PRIMARY KEY (model_run_id, focal_year)
        ON CONFLICT IGNORE
        )
        """
    )

    conn.close()

    return None
