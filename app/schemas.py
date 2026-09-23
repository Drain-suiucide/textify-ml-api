from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class PredictionResponse(BaseModel):
    status: str
    filename: str
    model: str
    predictions: list[Prediction]