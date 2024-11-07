import sys
if "." not in sys.path: sys.path.append(".")
import unittest
import logging
from core import Processor, Digit_Data_Module

logging.basicConfig(
        format="{asctime}::{levelname}::{name}::{message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

class Test_Data_Module(unittest.TestCase):
    def test_methods(self):
        data_module = Digit_Data_Module()
        print("***test_get_training_dataset***")
        training_dataset = data_module.get_training_dataset()
        train_dataset = training_dataset["train_dataset"]
        val_dataset = training_dataset["val_dataset"]
        print("---"*30)
    
        print("***test_get_inference_dataset***")
        test_dataset = data_module.get_inference_dataset()
        print("---"*30)
    
        print("***test_get_preprocessor***")
        preprocessor = data_module.get_preprocessor()
        print("---"*30)
if __name__=="__main__":
    unittest.main()