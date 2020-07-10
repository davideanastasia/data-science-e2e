from sklearn.pipeline import make_union, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklego.preprocessing import ColumnSelector

from this_project.preprocessing.crossfeature import CrossFeatureCalculator

from .datasets import CAT_COLS, NUM_COLS


def make_linear_preprocessor():
    return make_union(
        make_pipeline(
            ColumnSelector(CAT_COLS),
            SimpleImputer(strategy="constant", fill_value="MISSING"),
            OneHotEncoder(handle_unknown="ignore"),
        ),
        make_pipeline(
            ColumnSelector(NUM_COLS), SimpleImputer(strategy="median"), StandardScaler()
        ),
        make_pipeline(
            ColumnSelector(["occupation", "sex", "race", "marital_status"]),
            SimpleImputer(strategy="constant", fill_value="MISSING"),
            CrossFeatureCalculator(),
            OneHotEncoder(handle_unknown="ignore"),
        ),
    )
