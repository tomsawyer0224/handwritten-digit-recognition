from sklearn.utils import Bunch
from sklearn.datasets import fetch_openml

import logging

logger = logging.getLogger(__name__)
def load_dataset() -> Bunch:
    """
    This function will load MNIST digit dataset from openml
    """
    raw_dataset = fetch_openml(
        name="mnist_784",
        version=1,
        return_X_y=False,
        as_frame=True,
        data_home="./data"
    )
    logger.info(
        f"The MNIST Digit Dataset has been loaded with {len(raw_dataset["target"])} data points"
    )
    return raw_dataset
