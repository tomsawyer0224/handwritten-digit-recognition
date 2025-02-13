from matplotlib import pyplot as plt
from sklearn.utils import Bunch
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from typing import Union, Tuple


def visualize_classification_report(
    y_true: Union[np.ndarray, pd.DataFrame], y_pred: Union[np.ndarray, pd.DataFrame]
) -> str:
    report = classification_report(y_true=y_true, y_pred=y_pred)
    return report


def visualize_confusion_matrix(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    name: str = "",
) -> plt.figure:
    cmd = ConfusionMatrixDisplay.from_predictions(
        y_true=y_true, y_pred=y_pred, normalize="true", values_format=".2f"
    )
    fig = cmd.figure_
    fig.suptitle(name)
    return cmd.figure_


def visualize_image(
    dataset: Union[Bunch, pd.DataFrame, np.ndarray],
    prediction: Union[np.ndarray, pd.DataFrame] = None,
    nrows: int = 4,
    ncols: int = 4,
    figsize: Tuple[int] = (10, 10),
    name: str = "",
) -> plt.figure:
    """
    visualizes images from dataset
    """
    if isinstance(dataset, Bunch):
        data = np.array(dataset["data"])
        target = np.array(dataset["target"])
    else:
        data = np.array(dataset)
        target = None
    N = len(data)
    n_samples = nrows * ncols
    random_indices = np.random.permutation(N)
    indice_sample = random_indices[:n_samples]
    data_sample = data[indice_sample].reshape(-1, 28, 28)
    if target is not None:
        target_sample = target[indice_sample]
    else:
        target_sample = np.array([""] * n_samples)
    if prediction is not None:
        prediction = np.array(prediction)
        pred_sample = prediction[indice_sample]
        # pred_sample = [f"pred: {ps}/" for ps in pred_sample]
    else:
        pred_sample = [""] * n_samples
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    pr = "pred:" if prediction is not None else ""
    lb = "label:" if target is not None else ""
    slash = "/" if prediction is not None and target is not None else ""
    for i in range(nrows):
        for j in range(ncols):
            ax[i, j].set_axis_off()
            ax[i, j].imshow(data_sample[(i + 1) * (j + 1) - 1], cmap="gray")
            title = f"{pr} {pred_sample[(i+1)*(j+1) - 1]}{slash}{lb} {target_sample[(i+1)*(j+1) - 1]}"
            ax[i, j].set_title(title)
    fig.suptitle(name)
    return fig
