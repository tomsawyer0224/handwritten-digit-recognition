import sys
if "." not in sys.path: sys.path.append(".")
import unittest
import logging

from utils import visualize_image, visualize_confusion_matrix, visualize_classification_report
from core import Digit_Data_Module

logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
logger = logging.getLogger(__name__)

logger.info("prepare digit data module")
data_module = Digit_Data_Module()
val_dataset = data_module.val_dataset

class Test_visualize(unittest.TestCase):
    def test_visualize(self):
        fig = visualize_image(
            dataset=val_dataset,
            prediction=None, #val_dataset["target"],
            nrows=4,
            ncols=4,
            figsize=(8,8),
            name="validation dataset"
        )
        fig.savefig("./utils/test/image.png")
    def test_visualize_confusion_matrix(self):
        y_true = val_dataset["target"]
        y_pred = val_dataset["target"]
        cmd = visualize_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            name="confusion matrix"
        )
        cmd.savefig("./utils/test/cm.png")
    def test_visualize_classification_report(self):
        y_true = val_dataset["target"]
        y_pred = val_dataset["target"]
        report = visualize_classification_report(
            y_true=y_true,
            y_pred=y_pred
        )
        print(report)
if __name__=="__main__":
    unittest.main()