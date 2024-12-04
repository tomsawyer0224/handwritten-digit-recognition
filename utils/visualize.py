from matplotlib import pyplot as plt
from sklearn.utils import Bunch
import numpy as np
import pandas as pd
from typing import Union
def visualize_image(
        dataset: Bunch,
        prediction: Union[np.ndarray, pd.DataFrame] = None,
        nrows=4,
        ncols=4,
        figsize=(10,10)
    ) -> plt.figure:
    """
    visualizes images from dataset
    """
    data = np.array(dataset["data"])
    target = np.array(dataset["target"])
    N = len(target)
    n_samples = nrows*ncols
    random_indices = np.random.permutation(N)
    indice_sample = random_indices[:n_samples]
    data_sample = data[indice_sample].reshape(-1, 28, 28)
    target_sample = target[indice_sample]
    if prediction is not None:
        prediction = np.array(prediction)
        pred_sample = prediction[indice_sample]
        pred_sample = [f"pred: {ps}/" for ps in pred_sample]
    else:
        pred_sample = [""]*n_samples
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].set_axis_off()
            ax[i,j].imshow(data_sample[(i+1)*(j+1) - 1], cmap = "gray")
            title = f"{pred_sample[(i+1)*(j+1) - 1]}gt: {target_sample[(i+1)*(j+1) - 1]}"
            ax[i,j].set_title(title)
    return fig

