# Handwritten Digit Recognition Model
Build a model that can recognize the handwritten number.
# About project
- This project's goal was to demonstrate the entire machine learning process, including dataset preparation, training, fine-tuning, and model deployment.
- Optuna is used to optimize the hyperparameters of scikit-learn, XGBoost, LightGBM, and Catboost models on the handwritten digit dataset. MLflow is used to track the training process and deploy the final model to Docker.
# How it works
- Models are adjusted over hyperparameter space, which is defined via a configuration file.
- Every model is trained using the default set of parameters.
- Compare all of the results and return the best model.
> [!Note]
> [PyCaret](https://pycaret.org/) is another great library that can be used.
# Experiment
- Number of models: 8.
- Number of trials per model: 10.
- The winner: LGBMClassifier with 98.37% accuracy on the validation set.
# How to use
0. First, you should install [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html), [pyenv](https://github.com/pyenv/pyenv) and [Docker](https://docs.docker.com/engine/install/ubuntu/) on Ubuntu.
1. Open a Terminal (Terminal 1) and clone this repository:
```
git clone https://github.com/tomsawyer0224/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```
2. Run the following commands in Terminal 1 to build a virtual environment, install prerequisites, and generate scripts:
```
source init.sh
python run.py prepare
```
3. In the handwritten-digit-recognition directory, open a new Terminal (Terminal 2) and run command:
```
source scripts/start_tracking_server.sh
```
4. Change the model parameters as needed in config/project_config.yaml (you can add more models). Run the following command in Terminal 1 to optimize hyperparameters, train, and record the optimal model:
```
python run.py tune
```
This procedure can be repeated until we get the best model.

5. To view the results, open address 'tracking_uri' (specified in config/project_config.yaml, for example: http://127.0.0.1:8000) in a web browser.
   
6. After we've completed the tuning and training processes, run the below command in Terminal 1 to deploy the best mode to Docker:
```
python run.py deploy
```
7. To test the model the deployed model, open a new Terminal (Terminal 3) in the handwritten-digit-recognition directory and run command:
```
bash scripts/start_docker_container.sh
```
In Terminal 1, run command:
```
python inference.py
```
This script is used for testing purpose, follow the mlflow's instruction (on web browser) for inferrence.
> [!Note]
> To stop the tracking server, running this command in Terminal 1:
> ```
> bash scripts/stop_tracking_server.sh
> ```
> To stop the container, running this command in Terminal 1:
> ```
> bash scripts/stop_docker_container.sh
> ```
