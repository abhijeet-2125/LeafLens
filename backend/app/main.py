import logging
from contextlib import asynccontextmanager
from io import BytesIO
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from backend.app.schemas.prediction import PredictionResponse
from ml.serving.predictor import LeafPredictor

MAX_UPLOAD_SIZE = 10 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction model once during application startup."""
    logger.info("Loading LeafLens prediction model...")
    app.state.predictor = LeafPredictor()
    logger.info("Model loaded successfully.")
    yield
    logger.info("LeafLens API shutting down.")


app = FastAPI(
    title="LeafLens API",
    description="AI-powered plant leaf disease detection and monitoring system",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Return basic application information."""
    return {
        "application": "LeafLens",
        "version": "0.1.0",
        "message": "Welcome to LeafLens API",
    }


@app.get("/health", tags=["Operations"])
def health_check() -> dict[str, str]:
    """Check whether the API application is responding."""
    return {
        "status": "healthy",
        "service": "leaflens-api",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
async def predict_leaf(
    request: Request,
    file: UploadFile = File(...),
) -> PredictionResponse:
    """Predict the disease class from an uploaded leaf image."""

    contents = await file.read(MAX_UPLOAD_SIZE + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            image.verify()

        with Image.open(BytesIO(contents)) as image:
            image = image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid or supported image.",
        ) from None

    try:
        result = request.app.state.predictor.predict(image, top_k=5)

        logger.info(
            "Prediction completed: class=%s confidence=%.4f",
            result["predicted_class"],
            result["confidence"],
        )

        return PredictionResponse(
            **result,
            inference_device=str(request.app.state.predictor.device),
        )

    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction.",
        ) from None
