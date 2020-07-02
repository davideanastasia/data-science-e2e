import click

from hyperopt import fmin, hp, tpe, rand

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from this_project.censusdata import fetch_censusdata, make_linear_preprocessor


def train():
    def build_eval_fn(X, y):
        def eval_fn(params):
            # unpack parameters
            (C, ) = params

            # create train/test splits
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # pipeline
            pp = make_pipeline(
                make_linear_preprocessor(),
                LogisticRegression(C=C, class_weight='balanced', max_iter=1000)
            )

            # training
            pp.fit(X_train, y_train)
            
            # return score
            return pp.score(X_test, y_test)


        return eval_fn

    X, y = fetch_censusdata()
    print(X.shape, y.shape)

    space = [
        hp.quniform("C", 1.0, 25.0, 1.0),
    ]

    best = fmin(
        fn=build_eval_fn(X, y), space=space, algo=tpe.suggest, max_evals=5 
    )
    print(best)


if __name__ == "__main__":
    train()
