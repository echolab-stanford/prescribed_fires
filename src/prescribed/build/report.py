import pandas as pd


def report_treatments(treatments: pd.DataFrame) -> None:
    """Generate a cute report of the treatment data in the terminal"""

    if not isinstance(treatments, pd.DataFrame):
        raise ValueError("treatments should be a pandas DataFrame")

    print("\n Data report: Treatment dataset ")
    print("=================================")
    print(treatments.info())

    print("=================================")
    print("Counts per treatment type")
    print("=================================")

    print(
        treatments.groupby("Incid_Type", as_index=False).grid_id.count().to_markdown()
    )

    print("=================================")
    print("Counts per treatment type")
    print("=================================")

    print(treatments.groupby("treat", as_index=False).grid_id.nunique().to_markdown())

    print("=================================")
    print("Counts treatments")
    print("=================================")

    print(
        treatments.groupby("count_treats", as_index=False)
        .grid_id.nunique()
        .to_markdown()
    )

    return None
