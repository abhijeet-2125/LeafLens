#Prometheus metrics for LeafLens API and ML inference
from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter(
    "leaflens_http_requests_total",
    "Total HTTP requests received by LeafLens.",
    ["method", "endpoint", "status"])

HTTP_REQUEST_DURATION = Histogram(
    "leaflens_http_request_duration_seconds",
    "HTTP request processing duration in seconds.",
    ["method", "endpoint"])

PREDICTIONS_TOTAL = Counter(
    "leaflens_predictions_total",
    "Total successful leaf disease predictions.",
    ["predicted_class"])

PREDICTION_CONFIDENCE = Histogram(
    "leaflens_prediction_confidence",
    "Distribution of prediction confidence scores.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))

PREDICTION_ERRORS = Counter("leaflens_prediction_errors_total","Total model inference failures.")
