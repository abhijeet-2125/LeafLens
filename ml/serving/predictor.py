"""Model loading and inference service for LeafLens."""
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from ml.preprocessing.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
)
from ml.training.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts/models/mobilenetv3_best.pth"


class LeafPredictor:
    """Load the trained model and generate leaf disease predictions."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        self.idx_to_label = checkpoint["idx_to_label"]

        self.model = build_model(
            num_classes=len(self.idx_to_label),
            pretrained=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.inference_mode()
    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        """Predict the leaf disease and return the top-k class probabilities."""

        if not 1 <= top_k <= len(self.idx_to_label):
            raise ValueError(
                f"top_k must be between 1 and {len(self.idx_to_label)}."
            )

        image = image.convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)

        scores, indices = torch.topk(probabilities, k=top_k, dim=1)

        predictions = [
            {
                "class_name": self.idx_to_label[int(index)],
                "confidence": float(score),
            }
            for score, index in zip(
                scores[0].cpu().tolist(),
                indices[0].cpu().tolist(),
            )
        ]

        return {
            "predicted_class": predictions[0]["class_name"],
            "confidence": predictions[0]["confidence"],
            "top_predictions": predictions,
        }
