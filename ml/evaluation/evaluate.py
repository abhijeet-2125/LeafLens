import torch
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score)
from ml.preprocessing.dataset import build_dataloaders
from ml.training.config import BEST_MODEL_PATH, DEVICE
from ml.training.model import build_model
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
    device = torch.device(DEVICE)
    print("Loading test dataset...")
    dataloaders, label_to_idx = build_dataloaders()
    test_loader = dataloaders["test"]
    idx_to_label = {index: label for label, index in label_to_idx.items()}

    print("Loading trained checkpoint...")
    checkpoint = torch.load(BEST_MODEL_PATH,map_location=device, weights_only=False)
    model = build_model(num_classes=len(label_to_idx),pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint validation accuracy: {checkpoint['val_accuracy']:.4f}")

    all_predictions = []
    all_labels = []
    all_image_paths = []
    print("\nEvaluating test dataset...")
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())
            batch_start = batch_index * test_loader.batch_size
            batch_end = batch_start + len(labels)
            all_image_paths.extend(test_loader.dataset.dataframe.iloc[batch_start:batch_end]["filepath"].tolist())
            if (batch_index + 1) % 50 == 0:
                print(f"Processed {batch_index + 1}/{len(test_loader)} batches")
    accuracy = accuracy_score(all_labels,all_predictions)
    macro_precision = precision_score(all_labels, all_predictions,average="macro",zero_division=0)
    macro_recall = recall_score(all_labels,all_predictions,average="macro",zero_division=0)
    macro_f1 = f1_score(all_labels,all_predictions,average="macro",zero_division=0)

    print("LEAFLENS : TEST EVALUATION RESULTS")
    print(f"Test samples: {len(all_labels)}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1-score: {macro_f1:.4f}")

    print("\nCLASSIFICATION REPORT\n")
    class_names = [idx_to_label[index] for index in range(len(idx_to_label))]
    print(classification_report(all_labels,all_predictions,labels=list(range(len(class_names))),target_names=class_names,zero_division=0))
    matrix = confusion_matrix(all_labels,all_predictions,labels=list(range(len(class_names))))
    print("Confusion matrix shape:", matrix.shape)
    print("Total test predictions:", matrix.sum())
    print("Correct predictions:", matrix.trace())
    print("Incorrect predictions:", matrix.sum() - matrix.trace())

    # Save evaluation artifacts.
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "artifacts/evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_df = pd.DataFrame(matrix,index=class_names,columns=class_names)
    matrix_df.to_csv(output_dir / "confusion_matrix.csv")
        # Save misclassified image details.
    misclassified = pd.DataFrame({
        "filepath": all_image_paths,
        "actual_label": [
            idx_to_label[label] for label in all_labels
        ],
        "predicted_label": [
            idx_to_label[prediction]
            for prediction in all_predictions
        ],
    })

    misclassified = misclassified[
        misclassified["actual_label"]
        != misclassified["predicted_label"]
    ]

    misclassified.to_csv(
        output_dir / "misclassified_samples.csv",
        index=False,
    )

    print(
        f"\nMisclassified samples saved: {len(misclassified)}"
    )

    #to generate confusion matrix heatmap.
    figure, axis = plt.subplots(figsize=(20, 18))
    image = axis.imshow(matrix, interpolation="nearest",cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(title="LeafLens - Confusion Matrix",xlabel="Predicted Class",ylabel="Actual Class")
    axis.set_xticks(range(len(class_names)))
    axis.set_yticks(range(len(class_names)))
    axis.set_xticklabels(class_names,rotation=90,fontsize=7)
    axis.set_yticklabels(class_names,fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png",dpi=300,bbox_inches="tight")
    plt.close(figure)
    print("\nEvaluation artifacts saved:")
    print(output_dir / "confusion_matrix.csv")
    print(output_dir / "confusion_matrix.png")

if __name__ == "__main__":
    main()