import click
import mlflow

from hyperopt import fmin, hp, tpe, rand

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline

from this_project.censusdata import fetch_censusdata, make_linear_preprocessor


@click.command(help="Perform hyperparameter search with Hyperopt library.")
@click.option(
    "--max-evals",
    type=click.INT,
    default=10,
    help="Maximum number of runs to evaluate.",
)
def train(max_evals):
    def build_eval_fn(X, y):
        def eval_fn(params):
            with mlflow.start_run(nested=True) as child_run:
                #  unpack parameters
                (C,) = params

                mlflow.log_params({"C": C})

                # create train/test splits
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

                # pipeline
                clf = make_pipeline(
                    make_linear_preprocessor(),
                    LogisticRegression(
                        C=C, class_weight="balanced", max_iter=1000, random_state=0
                    ),
                )
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)

                precision, recall, fscore, support = precision_recall_fscore_support(
                    y_test, y_pred, average="binary"
                )

                mlflow.log_metrics(
                    {"precision": precision, "recall": recall, "fscore": fscore}
                )

                # return score
                return -recall

        return eval_fn

    X, y = fetch_censusdata()
    print(X.shape, y.shape)

    space = [hp.quniform("C", 1.0, 100.0, 0.5)]

    with mlflow.start_run() as run:
        mlflow.log_param("max_evals", max_evals)

        best = fmin(
            fn=build_eval_fn(X, y), space=space, algo=tpe.suggest, max_evals=max_evals
        )
        print(best)


if __name__ == "__main__":
    train()
