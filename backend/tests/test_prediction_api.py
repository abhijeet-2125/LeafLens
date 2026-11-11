
"""Integration tests for the LeafLens prediction API."""

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app


class MockPredictor:
    """Lightweight predictor for API integration tests."""

    device = "cpu"

    def predict(self, image, top_k=5):
        predictions = [
            {
                "class_name": "Tomato___Late_blight",
                "confidence": 0.95,
            },
            {
                "class_name": "Tomato___Early_blight",
                "confidence": 0.02,
            },
            {
                "class_name": "Tomato___healthy",
                "confidence": 0.015,
            },
            {
                "class_name": "Potato___Late_blight",
                "confidence": 0.01,
            },
            {
                "class_name": "Pepper,_bell___healthy",
                "confidence": 0.005,
            },
        ]

        return {
            "predicted_class": predictions[0]["class_name"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions[:top_k],
        }


@pytest.fixture
def client(monkeypatch):
    """Start the API with a mocked predictor."""

    monkeypatch.setattr(
        "backend.app.main.LeafPredictor",
        MockPredictor,
    )

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_image():
    """Generate a synthetic RGB image in memory."""

    image = Image.new("RGB", (256, 256), color=(34, 139, 34))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    return buffer.getvalue()


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["application"] == "LeafLens"


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_successful_prediction(client, test_image):
    response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", test_image, "image/jpeg")},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_class"] == "Tomato___Late_blight"
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 5
    assert result["model_name"] == "MobileNetV3-Large"
    assert result["inference_device"] == "cpu"


def test_empty_upload(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_invalid_image(client):
    response = client.post(
        "/predict",
        files={"file": ("invalid.jpg", b"not an image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert "valid or supported image" in response.json()["detail"]


def test_oversized_upload(client):
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={"file": ("large.jpg", oversized_content, "image/jpeg")},
    )

    assert response.status_code == 413
    assert "10 MB" in response.json()["detail"]


def test_prometheus_metrics_endpoint(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    metrics = response.text

    assert "leaflens_http_requests_total" in metrics
    assert "leaflens_http_request_duration_seconds" in metrics
    assert "leaflens_predictions_total" in metrics
    assert "leaflens_prediction_confidence" in metrics
    assert "leaflens_prediction_errors_total" in metrics


def test_prediction_metrics_recorded(client, test_image):
    prediction_response = client.post(
        "/predict",
        files={"file": ("leaf.jpg", test_image, "image/jpeg")},
    )

    assert prediction_response.status_code == 200

    response = client.get("/metrics")

    assert response.status_code == 200

    metrics = response.text

    assert 'leaflens_predictions_total{predicted_class="' in metrics
    assert "leaflens_prediction_confidence_count" in metrics
    assert 'endpoint="/health"' in metrics or 'endpoint="/predict"' in metrics
