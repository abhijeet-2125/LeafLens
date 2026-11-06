from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data/processed/dataset_manifest.csv"

BATCH_SIZE = 32
IMAGE_SIZE = 224
RANDOM_SEED = 42

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class LeafDataset(Dataset):
    """PyTorch dataset for PlantVillage leaf images."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_idx: dict[str, int],
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image_path = PROJECT_ROOT / row["filepath"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        label = self.label_to_idx[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """Create separate training and evaluation transformations."""

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def build_dataloaders():
    """Create training, validation and testing DataLoaders."""

    dataframe = pd.read_csv(MANIFEST_PATH)

    classes = sorted(dataframe["label"].unique())
    label_to_idx = {
        label: index for index, label in enumerate(classes)
    }

    train_transform, eval_transform = get_transforms()

    datasets = {
        split: LeafDataset(
            dataframe[dataframe["split"] == split],
            label_to_idx,
            train_transform if split == "train" else eval_transform,
        )
        for split in ["train", "validation", "test"]
    }

    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
        for split, dataset in datasets.items()
    }

    return dataloaders, label_to_idx