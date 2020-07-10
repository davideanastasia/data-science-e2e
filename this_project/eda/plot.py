import matplotlib.pyplot as plt

from sklearn.metrics import auc, precision_recall_curve, roc_curve


def plot_roc_pr_curves(clf, X, y, fig=None, figsize=(18, 9)):
    if fig is None:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)

        ax0.set_title("Receiver Operating Characteristic")
        ax0.plot([0, 1], ls="--")
        # ax0.plot([0, 0], [1, 0], c=".7")
        ax0.plot([1, 1], c=".7")
        ax0.set_ylabel("True Positive Rate")
        ax0.set_xlabel("False Positive Rate")
        ax0.set_ylim(0, 1)
        ax0.set_xlim(0, 1)

        ax1.set_title("Precision-Recall Curve")
        ax1.plot([1, 0], ls="--")
        # ax1.plot([0, 1], [1, 0], c=".7")
        ax1.plot([1, 1], c=".7")
        ax1.set_ylabel("Recall")
        ax1.set_xlabel("Precision")
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, 1)
    else:
        (ax0, ax1) = fig.axes

    # Plot ROC curves
    y_pred = clf.predict_proba(X)[:, 1]
    false_positive_rate, true_positive_rate, _ = roc_curve(y, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    clf_name = type(clf).__name__
    ax0.plot(false_positive_rate, true_positive_rate, label=f"{clf_name} (AUC = {roc_auc:.3f})")

    precision, recall, _ = precision_recall_curve(y, y_pred)

    ax1.plot(recall, precision, label=f"{clf_name}")

    ax0.legend()
    ax1.legend()

    return fig
