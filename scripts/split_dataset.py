from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data/raw/plantvillage dataset/color"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"

RANDOM_STATE = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

#to collect image paths and their class labels
records = []
for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    for image_path in class_dir.iterdir():
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            records.append({"filepath": str(image_path.relative_to(PROJECT_ROOT)),"label": class_dir.name})
df = pd.DataFrame(records)

#to separate training data from remaining data
train_df, remaining_df = train_test_split(df,test_size=0.30,stratify=df["label"],random_state=RANDOM_STATE)

#to divide remaining data equally into validation and test
val_df, test_df = train_test_split(remaining_df,test_size=0.50,stratify=remaining_df["label"],random_state=RANDOM_STATE)

train_df = train_df.assign(split="train")
val_df = val_df.assign(split="validation")
test_df = test_df.assign(split="test")

manifest = pd.concat([train_df, val_df, test_df],ignore_index=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "dataset_manifest.csv"
manifest.to_csv(output_path, index=False)

print("DATASET SPLIT SUMMARY")
print(f"Total images: {len(manifest)}")
print(f"Training: {len(train_df)} ({len(train_df) / len(manifest):.2%})")
print(f"Validation: {len(val_df)} ({len(val_df) / len(manifest):.2%})")
print(f"Testing: {len(test_df)} ({len(test_df) / len(manifest):.2%})")
print(f"\nNumber of classes: {manifest['label'].nunique()}")
print("\nClass distribution by split:")
print(pd.crosstab(manifest["label"], manifest["split"]))
print(f"\nManifest saved to: {output_path}")