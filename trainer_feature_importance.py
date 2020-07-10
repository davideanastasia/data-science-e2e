import click
import mlflow

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from this_project.censusdata import (
    fetch_censusdata,
    make_nonlinear_preprocessor,
    make_nonlinear_to_linear_preprocessor,
)
from this_project.censusdata.datasets import NUM_COLS, CAT_COLS

FEATURES = (
    CAT_COLS
    + NUM_COLS
    + [
        "occupation_cf_sex",
        "occupation_cf_race",
        "occupation_cf_marital_status",
        "sex_cf_race",
        "sex_cf_marital_status",
        "race_cf_marital_status",
    ]
)


@click.command(help="Run one-shot training of baseline model")
@click.option(
    "--regularisation",
    type=click.INT,
    default=27,
    help="Inverse Regularisation Strength",
)
@click.option(
    "--max-iter", type=click.INT, default=1000, help="Max number of iterations",
)
def trainer(regularisation: int, max_iter: int):
    with mlflow.start_run() as _:
        mlflow.set_tags({"training_type": "FeatureImportance"})
        mlflow.log_params({"C": regularisation, "max_iter": max_iter})

        X, y = fetch_censusdata()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        p1 = make_nonlinear_preprocessor()
        p2 = make_nonlinear_to_linear_preprocessor()
        clf = LogisticRegression(
            C=regularisation, max_iter=max_iter, class_weight="balanced", random_state=0
        )

        pp = make_pipeline(p1, p2, clf)

        pp.fit(X_train, y_train)

        # Log reference metrics
        y_pred = pp.predict(X_test)

        ref_precision, ref_recall, ref_fscore, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )

        mlflow.log_metrics(
            {"precision": ref_precision, "recall": ref_recall, "fscore": ref_fscore}
        )

        for i in range(len(FEATURES)):
            with mlflow.start_run(nested=True) as _:
                mlflow.set_tags({"training_type": "FeatureImportance"})
                mlflow.log_params({"feature": FEATURES[i]})

                X_test_tr = p1.transform(X_test)

                #  shuffle feature i
                indexes = np.arange(X_test_tr.shape[0])
                np.random.shuffle(indexes)
                X_test_tr[:, i] = X_test_tr[indexes, i]

                X_test_tr = p2.transform(X_test_tr)
                y_pred = clf.predict(X_test_tr)

                precision, recall, fscore, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="binary"
                )

                mlflow.log_metrics(
                    {
                        "precision": precision,
                        "precision_penalty": precision - ref_precision,
                        "recall": recall,
                        "recall_penalty": recall - ref_recall,
                        "fscore": fscore,
                        "fscore_penalty": fscore - ref_fscore,
                    }
                )


if __name__ == "__main__":
    trainer()
