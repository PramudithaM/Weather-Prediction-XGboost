# 🌦️ Rain Tomorrow Predictor

An end-to-end machine learning web app that predicts whether it will rain
tomorrow, based on today's weather conditions. Built as a hands-on learning
project covering the full ML lifecycle — from raw data to a deployed,
working product.

**Live demo:** _add your deployed link here_

<img width="1283" height="1037" alt="Screenshot 2026-09-29 164950" src="https://github.com/user-attachments/assets/012fac7c-4c05-414a-b6d3-23de1e789589" />


## 🧠 What this project covers

- Data cleaning & exploratory data analysis on 145K+ rows of real weather
  data ([Rain in Australia, Kaggle](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package))
- Feature engineering: missing-value imputation, one-hot encoding, date/season features
- Correct **time-ordered** train/test split (no data leakage)
- Model comparison: Logistic Regression, Random Forest, and XGBoost —
  evaluated on precision/recall/F1, not just accuracy, due to class imbalance
- Model serving via a FastAPI backend
- A Next.js + Tailwind frontend with a live probability gauge and a
  dynamic rain/sun visual effect based on the prediction

## 🏗️ Tech Stack

| Layer      | Tech |
|------------|------|
| ML         | Python, pandas, scikit-learn, XGBoost |
| Backend    | FastAPI, Pydantic, Uvicorn |
| Frontend   | Next.js (App Router), TypeScript, Tailwind CSS, Recharts |

## 📊 Model Comparison

| Model | Accuracy | F1 (Rain class) | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 84.3% | 0.573 | 0.66 | 0.51 |
| Random Forest | 84.9% | 0.520 | 0.76 | 0.39 |
| **XGBoost (chosen)** | 79.0% | **0.598** | 0.50 | **0.75** |

XGBoost was chosen as the production model despite a lower accuracy,
because for a rain forecast a **missed rain warning (false negative)**
is costlier than a false alarm — so recall matters more than raw accuracy.

## ⚠️ Known limitation

The model is trained on **Australian** weather data. Climate patterns
(humidity ranges, monsoon behavior, temperature bands) differ significantly
in other regions, so predictions are not reliable for, e.g., Sri Lankan
weather. This is a deliberate scope decision for this learning project —
see the "Future improvements" section below.

## 🚀 Running locally

### Backend
```bash
cd backend
conda activate weather_prediction   # or your venv
uvicorn app.main:app --reload
```
Runs at `http://127.0.0.1:8000` — interactive docs at `/docs`.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Runs at `http://localhost:3000`.

## 🔮 Future improvements

- Retrain on region-specific (e.g. Sri Lankan) weather data
- Hyperparameter tuning (GridSearchCV / Optuna)
- Feature importance analysis
- Dockerize both services

## 📁 Project Structure

```
weather-prediction-app/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── model/
│   │       ├── rain_model.json
│   │       └── model_columns.pkl
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx
│   │   └── layout.tsx
│   └── package.json
└── README.md
```

---
Built by Pramuditha as a personal learning project — not a class assignment.
