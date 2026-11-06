import torch

from ml.preprocessing.dataset import build_dataloaders
from ml.training.config import DEVICE
from ml.training.engine import set_seed, train_model
from ml.training.model import build_model


def main():
    set_seed()

    print("=" * 55)
    print("LEAFLENS - MOBILENETV3 TRAINING")
    print("=" * 55)

    print(f"Training device: {DEVICE}")

    dataloaders, label_to_idx = build_dataloaders()

    print(f"Number of classes: {len(label_to_idx)}")
    print(f"Training batches: {len(dataloaders['train'])}")
    print(f"Validation batches: {len(dataloaders['validation'])}")

    model = build_model(
        num_classes=len(label_to_idx),
        pretrained=True,
    )

    model, history = train_model(model,dataloaders["train"],dataloaders["validation"],label_to_idx)

    print("\nTraining completed.")
    print(f"Epochs completed: {len(history)}")


if __name__ == "__main__":
    main()