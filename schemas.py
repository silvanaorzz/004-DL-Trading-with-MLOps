# schemas.py
from typing import List, Dict, Optional
from pydantic import BaseModel


class PricePoint(BaseModel):
    Date: Optional[str]  # optional: can be provided or omitted
    Open: float
    High: float
    Low: float
    Close: float
    Volume: float


class PredictRequest(BaseModel):
    """
    `data` should be a list of candle dictionaries (most recent last).
    Minimum length should match the model timesteps/lookback window (default 30).
    """
    data: List[PricePoint]
    timesteps: Optional[int] = None  # optional override of model default lookback


class PredictResponse(BaseModel):
    prediction: int  # -1 (short), 0 (hold), 1 (long)
    confidence: float
    class_probs: Dict[str, float]
    model_name: str
    timesteps_used: int