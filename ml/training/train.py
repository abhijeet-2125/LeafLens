import json
from pathlib import Path
from ml.preprocessing.dataset import build_dataloaders
from ml.training.config import (
    BEST_MODEL_PATH,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    WEIGHT_DECAY)
from ml.training.engine import set_seed, train_model
from ml.training.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = PROJECT_ROOT / "artifacts/metrics/training_metrics.json"


def main():
    set_seed(RANDOM_SEED)
    print("LEAFLENS - MOBILENETV3 TRAINING")
    print(f"Training device: {DEVICE}")

    dataloaders, label_to_idx = build_dataloaders()
    print(f"Number of classes: {len(label_to_idx)}")
    print(f"Training batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['validation'])}")

    model = build_model(num_classes=len(label_to_idx),pretrained=True)
    _, history = train_model(model,dataloaders["train"],dataloaders["validation"],label_to_idx)
    if not history:
        raise RuntimeError("Training completed without producing epoch metrics.")
    best_epoch = max(history, key=lambda item: item["val_accuracy"])
    metrics = {
        "parameters": {
            "architecture": "MobileNetV3-Large",
            "num_classes": len(label_to_idx),
            "image_size": 224,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss_function": "CrossEntropyLoss",
            "random_seed": RANDOM_SEED,
            "device": DEVICE},
        "best_epoch": best_epoch["epoch"],
        "best_val_accuracy": best_epoch["val_accuracy"],
        "best_val_loss": best_epoch["val_loss"],
        "epochs_completed": len(history),
        "epoch_history": history,
        "checkpoint": str(BEST_MODEL_PATH.relative_to(PROJECT_ROOT))}

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nTraining completed.")
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Best validation accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
