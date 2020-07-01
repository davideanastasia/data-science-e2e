import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CrossFeatureCalculator(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self):
        pass

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        cols_num = X.shape[1]

        features = []
        for i in range(cols_num):
            for j in range(i, cols_num):
                if i != j:
                    features.append(X.iloc[:, i] + "/" + X.iloc[:, j])

        return np.vstack(features).T