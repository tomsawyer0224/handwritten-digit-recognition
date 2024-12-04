import sys
if "." not in sys.path: sys.path.append(".")
import unittest
import logging

from utils import visualize_image
from core import Digit_Data_Module

logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
logger = logging.getLogger(__name__)



class Test_visualize(unittest.TestCase):
    def test_visualize(self):
        logger.info("prepare digit data module")
        data_module = Digit_Data_Module()
        val_dataset = data_module.val_dataset

        fig = visualize_image(
            dataset=val_dataset,
            prediction=val_dataset["target"],
            nrows=4,
            ncols=4,
            figsize=(8,8)
        )
        fig.savefig("./utils/test/image.png")
if __name__=="__main__":
    unittest.main()