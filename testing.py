import torch
from torch.utils.data import DataLoader
import joblib
from pathlib import Path
from dataloader import test_loader

# 1. Load your pre-trained model and data
PROJECT_DIR = Path(__file__).resolve().parent
model = joblib.load(PROJECT_DIR / 'model/currentModel.pkl')

# 2. Set the model to evaluation mode
model.eval()

total_correct = 0
total_samples = 0

# 3. Disable gradient computation
with torch.no_grad():
    for inputs, labels in test_loader:
        # Forward pass only
        outputs = model(inputs)
        
        predictions = torch.argmax(outputs, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy * 100:.2f}%")