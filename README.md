# 📈 Demand Forecasting Dashboard

An end-to-end machine learning application for forecasting retail demand using classical time-series models and modern ML techniques — wrapped in a clean, interactive dashboard.

**Live Demo:**
Frontend: https://demand-forecasting-two.vercel.app/
Backend API: https://demand-forecasting-backend.onrender.com

---

## Overview

This project predicts weekly retail sales using multiple forecasting approaches and compares their performance in a unified dashboard.

It combines:

* Time-series modeling (ARIMA, SARIMA, Prophet)
* Machine learning (XGBoost)
* Model evaluation & comparison
* Full-stack deployment (FastAPI + React)

---

## Models Used

| Model   | Description                              |
| ------- | ---------------------------------------- |
| ARIMA   | Classical time-series model              |
| SARIMA  | Seasonal ARIMA capturing yearly patterns |
| Prophet | Trend + seasonality decomposition        |
| XGBoost | Supervised ML with engineered features   |

---

## Features

* Model comparison leaderboard (MAPE, MAE, RMSE)
* Predicted vs Actual visualization
* Model switching (ARIMA / SARIMA / Prophet / XGBoost)
* Metrics dashboard
* FastAPI backend serving predictions & metrics
* Fully deployed frontend (Vercel) and backend (Render)

---

## Results

| Model   | MAPE ↓ |
| ------- | ------ |
| XGBoost | ~2.6%  |
| SARIMA  | ~2.7%  |
| Prophet | ~3.2%  |
| ARIMA   | ~4.6%  |

👉 XGBoost performed best due to feature engineering and ability to capture non-linear patterns.

---

## Tech Stack

### ML / Data

* Python
* Pandas, NumPy
* Statsmodels (ARIMA, SARIMA)
* Prophet
* XGBoost
* Scikit-learn

### Backend

* FastAPI
* Uvicorn

### Frontend

* React (Vite)
* TailwindCSS
* Recharts

### Deployment

* Backend: Render
* Frontend: Vercel

---

## 📁 Project Structure

```
demand-forecasting/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── services.py
│   │   └── schemas.py
│
├── frontend/
│   ├── src/
│   └── index.html
│
├── artifacts/
│   ├── metrics/
│   ├── predictions/
│   └── results/
│
├── data/
│   └── raw/
│
└── README.md
```

---

## ⚙️ How to Run Locally

### 1️⃣ Clone repo

```
git clone https://github.com/Joydeeeep/demand-forecasting.git
cd demand-forecasting
```

---

### 2️⃣ Backend setup

```
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### 3️⃣ Frontend setup

```
cd frontend
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## Key Design Decisions

* **Artifacts instead of live model loading**
  Models are precomputed and stored as predictions & metrics for lightweight deployment.

* **Separation of concerns**
  Backend serves data, frontend handles visualization.

* **Time-series cross-validation**
  Used for realistic performance evaluation.

---

## Future Improvements

* Add feature importance visualization for XGBoost
* Add multi-store forecasting support
* Add confidence intervals for forecasts
* Add user input for custom predictions

---

## Author

**Joydeep Debsinha**

---

## ⭐ If you like this project

Give it a star ⭐ — it helps a lot!
