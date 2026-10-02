
# AQI Prediction Model

A Python project that trains a scikit-learn `RandomForestRegressor` on tabular data and serves predictions through both a Streamlit and a Gradio web interface. The repository is organised as a modular pipeline for data loading, preprocessing and model building, with `uv` for dependency management and DVC for data versioning.

![Python](https://img.shields.io/badge/Python-3.14%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Model](#machine-learning-model)
- [Application](#application)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Example Prediction Workflow](#example-prediction-workflow)
- [Author](#author)
- [License](#license)

---

## Project Overview

The project predicts Air Quality Index (AQI) values using a Random Forest regression model. The workflow is split into three modules under `src/`, orchestrated by `main.py`:

1. **Data loading** (`src/data_injection.py`): the `load()` function returns a pandas DataFrame.
2. **Preprocessing** (`src/data_preprocessing.py`): the `process()` function takes the DataFrame and returns `X_train`, `X_test`, `y_train`, `y_test`.
3. **Model building** (`src/model_build.py`): the `model()` function takes the train/test splits and returns test predictions together with an R² score.

The trained model is stored as a pickle file (`models/model.pkl`) and loaded by two separate applications: a Streamlit app and a Gradio app.

## Key Features

- Modular pipeline for data loading, preprocessing and model building
- Single entry point (`main.py`) for running the pipeline
- Random Forest regression model using scikit-learn
- Random Forest model configured with 300 trees
- Streamlit web interface for single-row predictions
- Gradio web interface for single-row predictions
- 12 numerical input features
- Dependency management with `uv`
- DVC configuration for data versioning

## Dataset

The project uses tabular data loaded through `src/data_injection.py`.

The model expects **12 numeric input features**. Both the Streamlit and Gradio applications currently use generic labels from `Feature 1` through `Feature 12`.

## Data Preprocessing

The `process()` function in `src/data_preprocessing.py` takes the loaded DataFrame and returns:

- `X_train`
- `X_test`
- `y_train`
- `y_test`

These outputs are then passed to the model-building function for training and evaluation.

## Machine Learning Model

| Item | Detail |
|---|---|
| Algorithm | `RandomForestRegressor` |
| Library | scikit-learn |
| Number of trees | 300 |
| Task | Regression |
| Input features | 12 numeric values |
| Serialized model | `models/model.pkl` |

The trained model is serialized using Pickle and loaded by the application interfaces for prediction.

## Application

The project provides two interfaces for generating predictions.

| | Streamlit (`app.py`) | Gradio (`gradio_app.py`) |
|---|---|---|
| Inputs | 12 numerical fields | 12 numerical fields |
| Model | Pre-trained Random Forest | Pre-trained Random Forest |
| Prediction | Single-row prediction | Single-row prediction |
| Output | Predicted value | Predicted value |

Both applications load the pre-trained model from:

```text
models/model.pkl
