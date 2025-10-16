# Sentiment Analysis Flask API

Simple Flask API that performs sentiment analysis on text using a Logistic Regression model trained with TF-IDF features.

## Overview
- Built with **Python, Flask, scikit-learn, joblib**
- Model: TF-IDF vectorizer + LogisticRegression
- Endpoint: `POST /predict` → returns sentiment ("Positive"/"Negative") and confidence

## Features
- Train script (`train_model.py`) builds `model.pkl` from `dataset.csv`
- Flask API (`app.py`) serves predictions locally
- Test client (`test_predict.py`) demonstrates API usage

## Setup (local)
1. `python -m venv venv`
2. `.\venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`
4. `python train_model.py`
5. `python app.py`  
API runs on **http://127.0.0.1:5001**

## Example Request
```bash
Invoke-RestMethod -Uri 'http://127.0.0.1:5001/predict' -Method Post -ContentType 'application/json' -Body '{"text":"I love this product"}'
