from pydantic import BaseModel
from typing import List, Dict, Any


class ForecastRequest(BaseModel):
    model_name: str


class ForecastResponse(BaseModel):
    model_name: str
    predictions: Dict[str, float]


class MetricsResponse(BaseModel):
    model_name: str
    metrics: Dict[str, Any]


class ModelsResponse(BaseModel):
    available_models: List[str]