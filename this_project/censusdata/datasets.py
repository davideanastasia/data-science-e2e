import pandas as pd

from this_project.datasets.common import (
    created_datasets_dir,
    fetch_asset,
    get_datasets_dir,
)

_BASE_SUBDIR = "censusdata"

_COL_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

CAT_COLS = [
    'workclass',
    'education',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native_country'
]

NUM_COLS = [
    'age',
    'fnlwgt',
    'capital_gain',
    'capital_loss',
    'hours_per_week'
]

TARGET_COL = "income"

_COL_DROP = [
    TARGET_COL,
    "education_num",
]

def fetch_censusdata():
    created_datasets_dir(_BASE_SUBDIR)

    fetch_asset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        _BASE_SUBDIR,
    )
    fetch_asset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
        _BASE_SUBDIR,
    )
    fetch_asset(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        _BASE_SUBDIR,
    )

    df = pd.read_csv(
        get_datasets_dir(_BASE_SUBDIR) + "/adult.data",
        names=_COL_NAMES,
        index_col=False,
    )

    return (
        df.drop(columns=_COL_DROP),
        df[TARGET_COL].map(lambda item: 1.0 if item.strip() == ">50K" else 0.0),
    )

