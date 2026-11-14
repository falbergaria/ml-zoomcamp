# Machine Learning Zoomcamp 2026 – Midterm Project

## Formula 1 Pit Stop Prediction

This project was developed as a midterm project for the [Machine Learning Zoomcamp 2026](https://courses.datatalks.club/ml-zoomcamp-2026/) course.

The dataset comes from the Kaggle competition [Predicting F1 Pit Stops](https://www.kaggle.com/competitions/playground-series-s6e5), where the data can be downloaded.

The goal is to build a machine learning model capable of predicting whether a Formula 1 driver will make a pit stop on the next lap, using information about tyre characteristics, race progression, driver position, lap performance, and other race-related variables.

## Dataset

The dataset contains information about Formula 1 drivers and their race conditions, with the target variable `PitNextLap` indicating whether a driver makes a pit stop on the following lap.

The original training and test datasets are not included in this repository because of their file sizes. To obtain them:

1. Visit the [Kaggle competition page](https://www.kaggle.com/competitions/playground-series-s6e5/data).
2. Log in to Kaggle and accept the competition rules, if required.
3. Download the competition data.
4. Extract the files and place `train.csv` and `test.csv` in the project directory.

The project expects the following structure:

```text
07-midterm-project/
├── train.csv
├── test.csv
├── notebook.ipynb
├── train.py
├── predict.py
├── predict-test.py
├── model.bin
├── Pipfile
├── Pipfile.lock
├── Dockerfile
└── README.md
```

The CSV files are not tracked by Git.

## Exploratory Data Analysis and Feature Engineering

The data was explored to understand the target variable, investigate feature distributions and relationships, and identify potentially useful predictors.

The preprocessing steps include:
- Normalizing column names and categorical values.
- Handling categorical features through one-hot encoding.
- Preparing the data for the classification algorithms.

The exploratory analysis and modelling experiments are documented in the Jupyter notebook.

## Model Development

Three classification algorithms were developed, tuned, and compared:

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

The models were evaluated using the ROC-AUC metric on a validation dataset. Hyperparameter tuning was performed to identify the best configuration among the tested alternatives.

The selected model was an XGBoost classifier with the following parameters:

| Parameter | Value |
|---|---:|
| `eta` | 0.3 |
| `max_depth` | 8 |
| `min_child_weight` | 20 |
| `num_boost_round` | 200 |

The final model achieved a validation ROC-AUC score of approximately **0.947**.

The notebook contains the complete experiments, model comparisons, hyperparameter tuning, and feature importance analysis.

## Training the Model

The final model can be trained using `train.py`.

Make sure that `train.csv` is present in the project directory, then run:

```bash
python train.py
```

This trains the XGBoost model using the full training dataset and saves the fitted model and feature vectorizer to `model.bin`.

## Prediction API

The trained model is served through a Flask API, using Waitress as the production WSGI server.

The API exposes a `/predict` endpoint that accepts a JSON object containing the features of a race observation and returns:
- `pit_probability`: the predicted probability of a pit stop on the next lap.
- `pitnextlap`: the corresponding binary prediction, using a probability threshold of 0.5.

### Running locally with Docker

Build the Docker image:

```bash
docker build -t pitnextlap .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 pitnextlap
```

The API will then be available at `http://localhost:9696/predict`.

### Testing the API

With the Docker container running, open another terminal and execute:

```bash
python predict-test.py
```

Alternatively, send a POST request to the endpoint. For example:

```python
import requests

url = 'http://localhost:9696/predict'

observation = {
    'Driver': 'PRO',
    'Compound': 'HARD',
    'Race': 'Bahrain Grand Prix',
    'Year': 2023,
    'PitStop': 0,
    'LapNumber': 32,
    'Stint': 3,
    'TyreLife': 12.0,
    'Position': 9,
    'LapTime (s)': 98.367,
    'LapTime_Delta': -0.0169999999999959,
    'Cumulative_Degradation': -21.056,
    'RaceProgress': 0.5614035087719298,
    'Position_Change': 1.0
}

response = requests.post(url, json=observation)
print(response.json())
```

Example response:

```json
{
    "pit_probability": 0.007743276190012693,
    "pitnextlap": false
}
```

## Project Structure

- `notebook.ipynb`: Exploratory data analysis, feature engineering, model training, evaluation, and hyperparameter tuning.
- `train.py`: Trains the final model using the full training dataset and exports the model artifact.
- `predict.py`: Flask application that serves predictions through a REST API.
- `predict-test.py`: Example script for testing the prediction endpoint.
- `model.bin`: Serialized XGBoost model and feature vectorizer.
- `Dockerfile`: Instructions for building the Docker image.
- `Pipfile` and `Pipfile.lock`: Python dependencies and their locked versions.

## Limitations

The dataset is synthetic and was created for the Kaggle Playground Series competition. Therefore, the validation performance should not be interpreted as a measure of how accurately the model would predict pit stops in real Formula 1 races.

