import sys
if "." not in sys.path: sys.path.append(".")
import unittest

from utils import get_model

class Test_get_model(unittest.TestCase):
    def test_get_model(self):
        sklearn_config = dict(
            model_class = "RandomForestClassifier",
            model_params = dict(
                n_estimators = 50,
                max_depth = 10
            )
        )
        print(get_model(sklearn_config, return_default_model=True))

        xgboost_config = dict(
            model_class = "XGBClassifier",
            model_params = dict(
                tree_method="hist",
                early_stopping_rounds=3
            )
        )
        print(get_model(xgboost_config, return_default_model=True))

        lightgbm_config = dict(
            model_class = "LGBMClassifier",
            model_params = dict(
                boosting_type="gbdt",
                max_depth=3
            )
        )
        print(get_model(lightgbm_config, return_default_model=True))

        catboost_config = dict(
            model_class = "CatBoostClassifier",
            model_params = dict(
                iterations=500,
                depth=8
            )
        )
        print(get_model(catboost_config, return_default_model=True))

if __name__ == "__main__":
    unittest.main()