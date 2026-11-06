import torch
import torch.nn as nn
from torch.optim import AdamW
from ml.preprocessing.dataset import build_dataloaders
from ml.training.model import build_model
device = torch.device("mps")
print("Loading one training batch...")

dataloaders, label_to_idx = build_dataloaders()
images, labels = next(iter(dataloaders["train"]))
images = images.to(device)
labels = labels.to(device)

print("Building model...")
model = build_model(num_classes=len(label_to_idx),pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.0001)

print("Running forward pass...")
model.train()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)

print("Running backward pass...")
loss.backward()
optimizer.step()

print("\nINTEGRATION TEST RESULTS")
print("Device:", device)
print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Loss:", round(loss.item(), 4))
print("Gradients computed:", any(parameter.grad is not None for parameter in model.parameters()))
print("Training step: SUCCESS")