import logging
from sklearn.utils import Bunch

from .data_loader import load_dataset
from .data_splitter import split_dataset
from core import Preprocessor

logger = logging.getLogger(__name__)

class Digit_Data_Module:
    """
    provides preprocessor, processed training dataset (includes train_dataset + val_dataset)
    and raw inference_dataset (raw testing dataset)
    """
    def __init__(self) -> None:
        raw_dataset = load_dataset()
        self.datasets = split_dataset(dataset=raw_dataset, rescale=True)
        self.preprocessor = Preprocessor()
    def get_training_dataset(self) -> Bunch:
        # train_dataset
        train_data = self.datasets["train_dataset"]["data"]
        train_data = self.preprocessor(train_data)
        train_target = self.datasets["train_dataset"]["target"]
        train_dataset = Bunch(data = train_data, target = train_target)
        logger.info(f"preprocessed the training dataset")

        # val_dataset
        val_data = self.datasets["val_dataset"]["data"]
        val_data = self.preprocessor(val_data)
        val_target = self.datasets["val_dataset"]["target"]
        val_dataset = Bunch(data = val_data, target = val_target)
        logger.info(f"preprocessed the validation dataset")

        return Bunch(
            train_dataset = train_dataset,
            val_dataset = val_dataset
        )
    def get_inference_dataset(self) -> Bunch:
        return self.datasets["test_dataset"]
    def get_preprocessor(self):
        return self.preprocessor
