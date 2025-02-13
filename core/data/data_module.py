import logging
from sklearn.utils import Bunch
import numpy as np
import pandas as pd

from .data_loader import load_dataset
from .data_splitter import split_dataset

logger = logging.getLogger(__name__)


class Digit_Data_Module:
    def __init__(self) -> None:
        raw_dataset = load_dataset()
        self.datasets = split_dataset(dataset=raw_dataset, rescale=True)

    @property
    def train_dataset(self):
        return self.datasets["train_dataset"]

    @property
    def val_dataset(self):
        return self.datasets["val_dataset"]

    @property
    def test_dataset(self):
        return self.datasets["test_dataset"]


class Toy_Data_Module:
    def __init__(self, n_samples=2048):
        rng = np.random.RandomState(seed=42)
        data = rng.rand(n_samples, 784)
        target = rng.randint(0, 10, (n_samples,))
        raw_datasets = Bunch(
            data=pd.DataFrame(data), target=pd.Series(target).astype(str)
        )
        self.datasets = split_dataset(dataset=raw_datasets, rescale=False)

    @property
    def train_dataset(self):
        return self.datasets["train_dataset"]

    @property
    def val_dataset(self):
        return self.datasets["val_dataset"]

    @property
    def test_dataset(self):
        return self.datasets["test_dataset"]
