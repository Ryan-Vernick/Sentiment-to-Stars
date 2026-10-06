from data_loader import load_data
from RNN_training import train_model, test_model
from joblib import dump

print("Loading data...")
train, test, _ = load_data()
print("Starting training...")
model = train_model(train_loader=train, output=True)
print("Testing model...")
test_model(model, test_loader=test, output=True)
print("Saving model...")
dump(model, 'currentModel.pkl')