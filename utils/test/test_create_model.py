import sys

if "." not in sys.path:
    sys.path.append(".")
import unittest

from utils import create_model


class Test_create_model(unittest.TestCase):
    def test_create_model(self):
        sklearn_config = dict(
            model_class="RandomForestClassifier",
            model_params=dict(n_estimators=50, max_depth=10),
        )
        print(create_model(sklearn_config))

        xgboost_config = dict(
            model_class="XGBClassifier",
            model_params=dict(tree_method="hist", early_stopping_rounds=3),
        )
        print(create_model(xgboost_config))

        lightgbm_config = dict(
            model_class="LGBMClassifier",
            model_params=dict(boosting_type="gbdt", max_depth=3),
        )
        print(create_model(lightgbm_config))

        catboost_config = dict(
            model_class="CatBoostClassifier", model_params=dict(iterations=500, depth=8)
        )
        print(create_model(catboost_config))


if __name__ == "__main__":
    unittest.main()
