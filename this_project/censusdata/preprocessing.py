from sklearn.pipeline import make_union, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from this_project.preprocessing.columnselector import ColumnSelector
from this_project.preprocessing.crossfeaturecalculator import CrossFeatureCalculator

from .datasets import CAT_COLS, NUM_COLS


def make_nonlinear_preprocessor():
    return make_union(
        make_pipeline(
            ColumnSelector(CAT_COLS),
            SimpleImputer(strategy="constant", fill_value="MISSING")
        ),
        make_pipeline(ColumnSelector(NUM_COLS)),
        make_pipeline(
            ColumnSelector(["occupation", "sex", "race", "marital_status"]),
            SimpleImputer(strategy="constant", fill_value="MISSING"),
            CrossFeatureCalculator()
        )
    )


def make_nonlinear_to_linear_preprocessor():
    return make_union(
        make_pipeline(
            ColumnSelector([0, 1, 2, 3, 4, 5, 6, 7]),
            OneHotEncoder(handle_unknown="ignore"),
        ),
        make_pipeline(
            ColumnSelector([8, 9, 10, 11, 12, 13]),
            SimpleImputer(strategy="median"),
            StandardScaler(),
        ),
        make_pipeline(
            ColumnSelector([14, 15, 16, 17, 18, 19]),
            OneHotEncoder(handle_unknown="ignore"),
        )
    )

def make_linear_preprocessor():
    return make_pipeline(
        make_nonlinear_preprocessor(),
        make_nonlinear_to_linear_preprocessor()
    )
