from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/raw/plantvillage dataset/color")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

class_counts = {}

for class_dir in sorted(DATASET_DIR.iterdir()):
    if class_dir.is_dir():
        image_count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        class_counts[class_dir.name] = image_count

print(f"Total classes: {len(class_counts)}")
print(f"Total images: {sum(class_counts.values())}")

print("\nClass-wise image distribution:\n")

for class_name, count in class_counts.items():
    print(f"{class_name:<45} {count:>5}")

counts = list(class_counts.values())

print("\nDistribution summary:")
print(f"Minimum images per class: {min(counts)}")
print(f"Maximum images per class: {max(counts)}")
print(f"Average images per class: {sum(counts) / len(counts):.2f}")