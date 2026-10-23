# Homework 5: Deploying Machine Learning Models

Solutions to Homework 5 of the [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

This homework covers model deployment, making predictions with a trained machine learning model, and interacting with a prediction API.

## Questions and Answers

| Question | Answer |
|---|---|
| Q1 | `uv 0.13.3` (`502f93ca6 2026-10-19 x86_64-pc-windows-msvc`) |
| Q2 | `1.7.2` |
| Q3 | `0.533` |
| Q4 | `0.770` |
| Q5 | `python:3.11.15-slim-bookworm` |
| Q6 | `0.770` |

## My Code

- **`question_3.py`** — Loads the serialized model and feature vectorizer from `pipeline.bin`, transforms a customer record, and prints the predicted probability.
- **`question_4.py`** — Sends a customer record to the local prediction API using an HTTP POST request and prints the response.

For the other questions, I followed the homework instructions and used code provided in the course repository where applicable.

## Running the Scripts

### Question 3

Make sure `pipeline.bin` is available in the working directory, then run:

```bash
python question_3.py
```

### Question 4

Start the prediction API according to the course instructions, ensuring it is listening at `http://localhost:9696/predict`. Then run:

```bash
python question_4.py
```

## Course Materials

The original homework instructions and supporting code are available in the [Machine Learning Zoomcamp repository](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/main/cohorts/2026/homework/05-deployment).
