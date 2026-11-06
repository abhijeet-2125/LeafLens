import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from ml.preprocessing.dataset import build_dataloaders
from ml.training.model import build_model

def main():
    device = torch.device("mps")
    dataloaders, label_to_idx = build_dataloaders()
    model = build_model(num_classes=len(label_to_idx),pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.0001)
    model.train()
    start_time = time.time()
    benchmark_batches = 20
    for batch_index, (images, labels) in enumerate(dataloaders["train"]):
        if batch_index >= benchmark_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if (batch_index + 1) % 5 == 0:
            print(f"Completed {batch_index + 1} batches")
    if device.type == "mps":
        torch.mps.synchronize()

    elapsed_time = time.time() - start_time
    seconds_per_batch = elapsed_time / benchmark_batches

    estimated_epoch_time = (seconds_per_batch * len(dataloaders["train"]))

    print("\nTRAINING BENCHMARK")
    print(f"Device: {device}")
    print(f"Batches completed: {benchmark_batches}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Time per batch: {seconds_per_batch:.2f} seconds")
    print(f"Estimated time per epoch: {estimated_epoch_time / 60:.2f} minutes")

if __name__ == "__main__":
    main()