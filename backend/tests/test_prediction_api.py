#Integration tests for the LeafLens prediction API
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture(scope="module")
def client():
    # to start the application once for this test module
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="module")
def test_image_path():
    manifest = pd.read_csv(PROJECT_ROOT / "data/processed/dataset_manifest.csv")
    sample = manifest[manifest["split"] == "test"].iloc[0]
    image_path = Path(sample["filepath"])
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    assert image_path.exists()
    return image_path

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["application"] == "LeafLens"

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_successful_prediction(client, test_image_path):
    with test_image_path.open("rb") as image:
        response = client.post("/predict",files={"file": (test_image_path.name,image,"image/jpeg")})
    assert response.status_code == 200
    result = response.json()
    assert result["predicted_class"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 5
    assert result["model_name"] == "MobileNetV3-Large"
    assert result["inference_device"] in {"mps", "cpu"}

def test_empty_upload(client):
    response = client.post("/predict",files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."

def test_invalid_image(client):
    response = client.post( "/predict", files={"file": ("invalid.jpg", b"not an image", "image/jpeg")})
    assert response.status_code == 400
    assert "valid or supported image" in response.json()["detail"]

def test_oversized_upload(client):
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)
    response = client.post("/predict",files={"file": ( "large.jpg",oversized_content,"image/jpeg")})
    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]
