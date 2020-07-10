import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

# TODO : to complete
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        # if the columns parameter is not a list, make it into a list
        self.columns = columns

    def fit(self, X, y=None):
        # self.columns_ = as_list(self.columns)
        # self._check_X_for_type(X)
        # self._check_column_length()
        # self._check_column_names(X)
        return self

    def transform(self, X):
        # self._check_X_for_type(X)
        if self.columns:
            if isinstance(X, pd.DataFrame):
                return X[self.columns]
            elif isinstance(X, np.ndarray):
                return X[:, self.columns]
        return X

    # def get_feature_names(self):
    #     return self.columns

    # def _check_column_length(self):
    #     """Check if no column is selected"""
    #     if len(self.columns_) == 0:
    #         raise ValueError(
    #             "Expected columns to be at least of length 1, found length of 0 instead"
    #         )

    # def _check_column_names(self, X):
    #     """Check if one or more of the columns provided doesn't exist in the input DataFrame"""
    #     non_existent_columns = set(self.columns_).difference(X.columns)
    #     if len(non_existent_columns) > 0:
    #         raise KeyError(f"{list(non_existent_columns)} column(s) not in DataFrame")

    # @staticmethod
    # def _check_X_for_type(X):
    #     """Checks if input of the Selector is of the required dtype"""
    #     if not isinstance(X, pd.DataFrame):
    #         raise TypeError("Provided variable X is not of type pandas.DataFrame")