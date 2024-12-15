import sys
if "." not in sys.path: sys.path.append(".")
import unittest
from utils import load_config

class Test_config_parser(unittest.TestCase):
    def test_load_config(self):
        config_file = "./config/sklearn_training_config.yaml"
        config = load_config(config_file=config_file)
        print(config)
if __name__=="__main__":
    unittest.main()