from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from backend.app.schemas import (
    ForecastRequest,
    ForecastResponse,
    MetricsResponse,
    ModelsResponse
)
from backend.app.services import ArtifactService

app = FastAPI(
    title="Demand Forecasting API",
    description="API for serving forecasting models and results",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://demand-forecasting-two.vercel.app",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Demand Forecasting API is running 🚀",
        "docs": "/docs"
    }
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running"}


@app.get("/models", response_model=ModelsResponse)
def get_models():
    models = ArtifactService.list_models()
    return {"available_models": models}


@app.get("/metrics/{model_name}", response_model=MetricsResponse)
def get_metrics(model_name: str):
    try:
        metrics = ArtifactService.load_metrics(model_name)
        return {"model_name": model_name, "metrics": metrics}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/predictions/{model_name}")
def get_predictions(model_name: str):
    try:
        preds = ArtifactService.load_predictions(model_name)
        return {
            "model_name": model_name,
            "predictions": {str(idx): float(val) for idx, val in preds.items()}
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/comparison")
def get_comparison():
    try:
        df = ArtifactService.load_comparison()
        return df.reset_index().rename(columns={"index": "model"}).to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/actuals")
def get_actuals():
    df = ArtifactService.load_actual_data()

    # Convert Date column properly
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Format to match predictions
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    return {
        "actuals": {
            row["Date"]: float(row["Weekly_Sales"])
            for _, row in df.iterrows()
        }
    }

@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest):
    try:
        preds = ArtifactService.load_predictions(request.model_name)
        return {
            "model_name": request.model_name,
            "predictions": {str(idx): float(val) for idx, val in preds.items()}
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

