import json
import logging
import time
from pathlib import Path
import pandas as pd
import ray
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage dataset" / "color"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ray_image_metadata.parquet"
METRICS_FILE = PROJECT_ROOT / "artifacts" / "metrics" / "ray_ingestion_metrics.json"

NUM_CPUS = 2
NUM_BLOCKS = 8
BATCH_SIZE = 128

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def inspect_batch(batch: pd.DataFrame) -> pd.DataFrame:
    results = []
    for record in batch.to_dict("records"):
        image_path = record["filepath"]
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
                image.verify()

            results.append({
                "filepath": image_path,
                "class_name": record["class_name"],
                "width": width,
                "height": height,
                "mode": mode,
                "file_size_bytes": Path(image_path).stat().st_size,
                "is_valid": True,
                "error": None})

        except Exception as exc:
            results.append({
                "filepath": image_path,
                "class_name": record["class_name"],
                "width": None,
                "height": None,
                "mode": None,
                "file_size_bytes": None,
                "is_valid": False,
                "error": type(exc).__name__})
    return pd.DataFrame(results)

def main() -> None:
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    records = [
        {"filepath": str(path), "class_name": path.parent.name}
        for path in sorted(DATASET_DIR.glob("*/*"))
        if path.is_file()]

    if not records:
        raise RuntimeError(f"No image files found in {DATASET_DIR}")

    logger.info("Discovered %s image files.", len(records))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ray.init(num_cpus=NUM_CPUS, include_dashboard=False)
    try:
        dataset = ray.data.from_items(records,override_num_blocks=NUM_BLOCKS)
        start_time = time.perf_counter()
        processed = dataset.map_batches(inspect_batch,batch_format="pandas",batch_size=BATCH_SIZE)
        metadata = processed.to_pandas()
        processing_seconds = time.perf_counter() - start_time
        metadata.to_parquet(OUTPUT_FILE, index=False)
        total_images = len(metadata)
        valid_images = int(metadata["is_valid"].sum())
        invalid_images = total_images - valid_images
        throughput = (total_images / processing_seconds if processing_seconds > 0 else 0.0)

        summary = {
            "framework": "Ray Data",
            "dataset": "PlantVillage RGB",
            "total_images": total_images,
            "valid_images": valid_images,
            "invalid_images": invalid_images,
            "num_classes": int(metadata["class_name"].nunique()),
            "num_cpus": NUM_CPUS,
            "num_blocks": NUM_BLOCKS,
            "batch_size": BATCH_SIZE,
            "processing_seconds": round(processing_seconds, 3),
            "throughput_images_per_second": round(throughput, 2),
            "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT))}

        with METRICS_FILE.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)

        logger.info("Ray Data ingestion completed.")
        logger.info("Summary: %s", json.dumps(summary, indent=2))
        logger.info("Parquet output: %s", OUTPUT_FILE)
        logger.info("Metrics output: %s", METRICS_FILE)

    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()
