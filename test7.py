from mlflow.models import convert_input_example_to_serving_input
import numpy as np
import pandas as pd

x = np.random.rand(2,3)
y = pd.DataFrame(x)
serving_input = convert_input_example_to_serving_input(y)
print(type(serving_input))
print(serving_input)
