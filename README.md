# LeafLens 🌿
### AI-Powered Plant Leaf Disease Detection with an MLOps and AIOps Pipeline

LeafLens is an end-to-end plant leaf disease detection system that uses deep learning to identify diseases from leaf images. It combines an image classification model with a reproducible ML pipeline, API-based inference, containerized deployment, and real-time monitoring.

The project is designed to demonstrate the complete lifecycle of an AI application — from dataset preparation and model training to deployment, inference, and operational monitoring.

**Author:** Abhijeet Kumar  
**Student ID:** DA24B001  
**Project:** LeafLens  
**Version:** 0.1.0

---

## 1. Problem Statement

Plant diseases can significantly affect agricultural productivity. Early identification helps farmers and agricultural practitioners take timely action.

LeafLens addresses this problem by providing an image-based disease classification system. A user uploads a photograph of a plant leaf, and the application returns the predicted disease class along with its confidence score.

Beyond model development, LeafLens incorporates MLOps and AIOps practices to make the application reproducible, deployable, testable, and observable.

## 2. Key Features

- **Deep Learning:** MobileNetV3-Large-based image classification.
- **38 Disease Classes:** Classification across 38 PlantVillage categories.
- **Data Management:** DVC-based dataset and pipeline tracking.
- **Distributed Data Processing:** Ray Data-based image ingestion.
- **Experiment Tracking:** MLflow experiment and metric logging.
- **REST API:** FastAPI-based prediction service.
- **User Interface:** Streamlit-based image upload and prediction interface.
- **Containerization:** Docker and Docker Compose.
- **Monitoring:** Prometheus metrics collection and Grafana dashboards.
- **Automated Testing:** Pytest-based API test suite.

## 3. System Architecture

The application consists of three major workflows:

1. **ML pipeline:** Dataset preparation, model training, evaluation, and experiment tracking.
2. **Inference pipeline:** Image upload, preprocessing, model inference, and prediction response.
3. **Monitoring pipeline:** Application metrics collection, Prometheus scraping, and Grafana visualization.

### High-Level Architecture

```text
             PLANTVILLAGE DATASET
                      |
                      v
              DVC DATA PIPELINE
                      |
          +-----------+-----------+
          |                       |
          v                       v
   Dataset Preparation       Ray Data Ingestion
          |                       |
          v                       v
   Train / Validation /       Metadata and
       Test Split             Ingestion Metrics
          |
          v
   MobileNetV3-Large
          |
          v
     Model Evaluation
          |
          v
      MLflow Tracking
          |
          v
    Trained Model Artifact
          |
          v
       FASTAPI BACKEND
          ^
          |
    STREAMLIT FRONTEND
          |
          v
     Disease Prediction
          |
          v
    Prometheus Metrics
          |
          v
      Grafana Dashboard
```

## 4. Dataset

LeafLens uses the PlantVillage dataset.

| Property | Value |
|---|---:|
| Total images | 54,305 |
| Number of classes | 38 |
| Training images | 38,013 |
| Validation images | 8,146 |
| Test images | 8,146 |
| Image format | RGB |
| Original image dimensions | 256 × 256 |

The dataset was divided using a stratified split with `random_state=42`, preserving class representation across the three subsets.

**Dataset source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

The dataset is distributed under the CC-BY-NC-SA-4.0 license. Refer to the dataset source for its usage conditions.

## 5. Model Development

LeafLens uses **MobileNetV3-Large**, pretrained on ImageNet, for multi-class leaf disease classification.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | MobileNetV3-Large |
| Pretraining | ImageNet |
| Input size | 224 × 224 |
| Batch size | 32 |
| Epochs | 5 |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Loss function | CrossEntropyLoss |

### Test Results

The final model was evaluated on the held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 99.53% |
| Macro Precision | 99.27% |
| Macro Recall | 99.34% |
| Macro F1-score | 99.28% |
| Correct predictions | 8,108 |
| Incorrect predictions | 38 |

These results represent performance on the PlantVillage test split. They should not be interpreted as equivalent performance on photographs collected under unrestricted field conditions.

## 6. MLOps Implementation

### 6.1 DVC — Data and Pipeline Versioning

DVC manages the machine learning workflow through three pipeline stages:

| Stage | Purpose |
|---|---|
| `prepare_dataset` | Generates the dataset split manifest |
| `train_model` | Trains the classification model |
| `ray_ingest` | Processes image data using Ray Data |

The pipeline is defined in `dvc.yaml`, with its dependency and output state recorded in `dvc.lock`.

Run the pipeline using:

```bash
dvc repro
```

The raw dataset must be available at the configured `data/raw/plantvillage dataset/color` path.

### 6.2 Ray Data — Distributed Data Ingestion

Ray Data is used to process and validate image records.

Recorded ingestion results:

| Metric | Result |
|---|---:|
| Images processed | 54,305 |
| Invalid images | 0 |
| CPU allocation | 2 |
| Data blocks | 8 |
| Batch size | 128 |
| Processing time | 7.553 seconds |
| Recorded throughput | 7,189.73 images/sec |

The processed metadata is stored at:

`data/processed/ray_image_metadata.parquet`

### 6.3 MLflow — Experiment Tracking

MLflow is used to track model training experiments, parameters, metrics, and model artifacts.

Tracked information includes:

- Model configuration and hyperparameters.
- Training and validation metrics.
- Best validation performance.
- Model checkpoint and associated metadata.

The configured experiment is named `LeafLens-PlantVillage`.

## 7. Application and Inference

### Backend — FastAPI

The backend provides REST endpoints for application health checks, image prediction, and metrics exposure.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Image prediction |
| GET | `/metrics` | Prometheus metrics |

The prediction pipeline performs image validation, RGB conversion, resizing, normalization, and model inference.

### Frontend — Streamlit

The Streamlit application provides a user-facing interface for uploading leaf images and viewing model predictions.

The frontend communicates with the backend through the configured API URL.

## 8. Testing

The backend test suite contains eight API tests covering:

- Root endpoint.
- Health endpoint.
- Successful prediction.
- Empty image upload.
- Invalid image upload.
- Oversized image upload.
- Prometheus metrics endpoint.
- Prediction metrics recording.

**Recorded test result:** 8 tests passed in 4.74 seconds.

Run the tests from the project root:

```bash
pytest backend/tests/ -v
```

## 9. Docker Deployment

LeafLens uses Docker Compose to run four services:

| Service | Host Port | Purpose |
|---|---:|---|
| Backend | 8001 | FastAPI inference |
| Frontend | 8501 | Streamlit interface |
| Prometheus | 9092 | Metrics collection |
| Grafana | 3002 | Monitoring dashboards |

### Prerequisites

- Docker Desktop
- Docker Compose
- Trained model checkpoint at `artifacts/models/mobilenetv3_best.pth`

### Build the Images

From the project root:

```bash
docker build -f backend/Dockerfile -t leaflens-backend:latest .
```

```bash
docker build -f frontend/Dockerfile -t leaflens-frontend:latest .
```

### Start the Application

```bash
docker compose up -d
```

Check service status:

```bash
docker compose ps
```

### Access the Services

| Service | URL |
|---|---|
| LeafLens Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8001 |
| Prometheus | http://localhost:9092 |
| Grafana | http://localhost:3002 |

The local Grafana demonstration credentials are:

- Username: `admin`
- Password: `leaflens_local_admin`

Change the default password before exposing the deployment beyond a local demonstration environment. You can override it using the `GRAFANA_ADMIN_PASSWORD` environment variable.

### Stop the Application

```bash
docker compose down
```

To remove the persistent monitoring volumes as well:

```bash
docker compose down -v
```

## 10. Monitoring and Observability

LeafLens integrates Prometheus and Grafana to monitor application behaviour.

The monitoring setup tracks:

- HTTP request counts.
- API request duration.
- Prediction counts by disease class.
- Prediction confidence distribution.
- Prediction failures.
- Backend availability and request latency.

The provisioned Grafana dashboard, **LeafLens | AIOps Monitoring**, contains eight monitoring panels covering application availability, request rates, prediction activity, latency, errors, and class-wise predictions.

## 11. Project Structure

```text
LeafLens/
│
├── artifacts/
│   ├── evaluation/
│   ├── metrics/
│   └── models/
│
├── backend/
│   ├── app/
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── architecture/
│   ├── design/
│   ├── testing/
│   └── user_manual/
│
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── ml/
│   ├── evaluation/
│   ├── preprocessing/
│   ├── serving/
│   └── training/
│
├── monitoring/
│   ├── prometheus/
│   └── grafana/
│
├── pipelines/
│   └── ray_ingestion.py
│
├── scripts/
│
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
└── pyproject.toml
```

## 12. Project Documentation

The five mandatory project documents are available in the `docs/` directory.

| Document | Location |
|---|---|
| Architecture Document | `docs/architecture/Architecture_Document.pdf` |
| High-Level Design | `docs/design/High_Level_Design.pdf` |
| Low-Level Design | `docs/design/Low_Level_Design.pdf` |
| Test Plan and Report | `docs/testing/Test_Plan_and_Report.pdf` |
| User Manual | `docs/user_manual/User_Manual.pdf` |

## 13. Limitations

- The model is trained and evaluated using the PlantVillage dataset, which contains relatively controlled image backgrounds.
- Real-world photographs may contain lighting variations, occlusions, complex backgrounds, and different disease stages.
- High test accuracy on the existing split does not establish equivalent field performance.
- The system provides a predicted class and confidence score, not a verified agricultural diagnosis.
- Further validation using field-collected images is required before practical agricultural deployment.

## 14. Future Improvements

- Evaluate performance on real-world field images.
- Improve robustness to lighting, background, and image-quality variations.
- Explore model explainability using Grad-CAM.
- Explore mobile-friendly deployment.

## 15. Acknowledgements

This project uses the PlantVillage dataset and open-source technologies including PyTorch, Torchvision, DVC, Ray, MLflow, FastAPI, Streamlit, Docker, Prometheus and Grafana.

---

**LeafLens — From leaf images to actionable disease predictions.**