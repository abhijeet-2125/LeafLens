#Pydantic schemas for LeafLens prediction responses
from pydantic import BaseModel, Field

class PredictionItem(BaseModel):
    #single disease prediction with its confidence score
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    #Structured response returned by the prediction API.
    predicted_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[PredictionItem]
    model_name: str = "MobileNetV3-Large"
    inference_device: str
