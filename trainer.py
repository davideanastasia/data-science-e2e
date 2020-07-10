import click
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from this_project.censusdata import fetch_censusdata, make_linear_preprocessor


@click.command(help="Run one-shot training of baseline model")
@click.option(
    "--regularisation",
    type=click.INT,
    default=27,
    help="Inverse Regularisation Strength",
)
@click.option(
    "--max-iter",
    type=click.INT,
    default=1000,
    help="Max number of iterations",
)
def trainer(regularisation: int, max_iter: int):
    with mlflow.start_run() as _:
        mlflow.set_tags({"training_type": "baseline"})
        mlflow.log_params({"C": regularisation, "max_iter": max_iter})

        X, y = fetch_censusdata()
        clf = make_pipeline(
            make_linear_preprocessor(),
            LogisticRegression(
                C=regularisation, max_iter=max_iter, class_weight="balanced", random_state=0
            ),
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )

        mlflow.log_metrics(
            {"precision": precision, "recall": recall, "fscore": fscore}
        )


if __name__ == "__main__":
    trainer()
