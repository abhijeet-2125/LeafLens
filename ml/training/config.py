from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_SEED = 42

# Dataset
NUM_CLASSES = 38
IMAGE_SIZE = 224
BATCH_SIZE = 32

# Training
EPOCHS = 5
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# Optimization
OPTIMIZER = "AdamW"
LOSS_FUNCTION = "CrossEntropyLoss"
DEVICE = "mps"
MODEL_DIR = PROJECT_ROOT / "artifacts/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "mobilenetv3_best.pth"