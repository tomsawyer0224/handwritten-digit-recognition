import logging
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split

from utils import name2id

logger = logging.getLogger(__name__)

def split_dataset(dataset: Bunch, rescale: bool = True) -> Bunch:
    """
    splits a dataset into three datasets: train, val, test dataset
    args:
        dataset: Bunch object with keys: data, target
        rescale: scales to [0.0, 1.0] range or not
    return:
        Bunch object with keys: train_dataset, val_dataset, test_dataset
    """
    if rescale:
        dataset["data"] = dataset["data"]/255.0
        #dataset["data"] = (dataset["data"]-127.5)/127.5
        logger.info("rescaled the dataset to [0.0, 1.0]")
        #logger.info(f"{dataset["data"].min().min()=}, {dataset["data"].max().max()=}")
    dataset["target"] = name2id(dataset["target"])
    train_data, test_data, train_target, test_target = train_test_split(
        dataset["data"],
        dataset["target"],
        test_size=0.1,
        random_state=42
    )
    train_data, val_data, train_target, val_target = train_test_split(
        train_data,
        train_target,
        test_size=0.15,
        random_state=42,
    )
    train_dataset = Bunch(data = train_data, target = train_target, size = len(train_target))
    val_dataset = Bunch(data = val_data, target = val_target, size = len(val_target))
    test_dataset = Bunch(data = test_data, target = test_target, size = len(test_target))
    logger.info(
        f"splitted the dataset into 3 datasets: "\
        f"train dataset of size {train_dataset['size']}, "\
        f"validation dataset of size {val_dataset['size']}, "\
        f"test dataset of size {test_dataset['size']}"
    )
    return Bunch(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        test_dataset = test_dataset
    )