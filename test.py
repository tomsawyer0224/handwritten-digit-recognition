from sklearn.datasets import fetch_openml

raw_dataset = fetch_openml(
        name="mnist_784",
        version=1,
        return_X_y=False,
        as_frame=True,
        data_home="./data"
    )
print(raw_dataset)