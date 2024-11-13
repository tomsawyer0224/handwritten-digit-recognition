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
    def print_dataset_info(self, dataset):
        data = dataset["data"]
        target = dataset["target"]
        print(f"data value:\n{data}")
        print(f"data type: {type(data)}")
        print(f"data range: [{data.min()}, {data.max()}]")
        print(f"target value:\n{target}")
        print(f"target type: {type(target)}")
        #print(f"target range: [{target.min()}, {target.max()}]")
    def test_methods(self):
        data_module = Digit_Data_Module()
        print("***test_get_training_dataset***")
        training_dataset = data_module.get_training_dataset()
        train_dataset = training_dataset["train_dataset"]
        val_dataset = training_dataset["val_dataset"]
        print("train_dataset")
        self.print_dataset_info(train_dataset)
        print("++"*30)
        print("val_dataset")
        self.print_dataset_info(val_dataset)
        print("---"*30)
    
        print("***test_get_inference_dataset***")
        test_dataset = data_module.get_inference_dataset()
        print("test_dataset")
        self.print_dataset_info(test_dataset)
        print("---"*30)
    
        print("***test_get_preprocessor***")
        preprocessor = data_module.get_preprocessor()
        print(preprocessor)
        print("---"*30)
if __name__=="__main__":
    unittest.main()