from collections import Counter
from pathlib import Path

from PIL import Image

DATASET_DIR = Path("data/raw/plantvillage dataset/color")

image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

total_images = 0
corrupted_images = []
image_modes = Counter()
image_sizes = Counter()

for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    for image_path in class_dir.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        total_images += 1

        try:
            with Image.open(image_path) as image:
                image.verify()

            with Image.open(image_path) as image:
                image_modes[image.mode] += 1
                image_sizes[image.size] += 1

        except Exception as error:
            corrupted_images.append(
                (str(image_path), str(error))
            )

print(f"Total images checked: {total_images}")
print(f"Corrupted images: {len(corrupted_images)}")

print("\nImage modes:")
for mode, count in image_modes.most_common():
    print(f"{mode}: {count}")

print("\nTop 10 image dimensions:")
for size, count in image_sizes.most_common(10):
    print(f"{size}: {count}")

if corrupted_images:
    print("\nCorrupted image examples:")
    for path, error in corrupted_images[:10]:
        print(path, error)