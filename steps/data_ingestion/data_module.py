import logging
from sklearn.utils import Bunch

from .data_loader import load_dataset
from .data_splitter import split_dataset
from .processor import Processor

logger = logging.getLogger(__name__)

class Digit_Data_Module:
    """
    provides preprocessor, processed training dataset (includes train_dataset + val_dataset)
    and raw inference_dataset (raw testing dataset)
    """
    def __init__(self) -> None:
        raw_dataset = load_dataset()
        self.datasets = split_dataset(raw_dataset)
        self.preprocessor = Processor()
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
    
if __name__=="__main__":
    logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    ds = Digit_Data_Module()
    training_dataset = ds.get_training_dataset()
    train_dataset = training_dataset["train_dataset"]
    val_dataset = training_dataset["val_dataset"]
    test_dataset = ds.get_inference_dataset()