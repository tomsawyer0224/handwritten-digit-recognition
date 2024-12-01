from sklearn.utils import Bunch
from sklearn.datasets import fetch_openml

import logging

logger = logging.getLogger(__name__)
def load_dataset() -> Bunch:
    """
    loads MNIST digit dataset from openml
    """
    raw_dataset = fetch_openml(
        name="mnist_784",
        version=1,
        return_X_y=False,
        as_frame=True,
        #data_home="./core/data/digit_data",
        data_home="./data"
    )
    logger.info(f"loaded MNIST Digit Dataset with {len(raw_dataset["target"])} data points")
    return raw_dataset
