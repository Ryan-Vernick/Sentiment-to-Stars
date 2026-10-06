from transformers import AutoModelForSequenceClassification
from joblib import dump

model = AutoModelForSequenceClassification.from_pretrained("model")

dump(model, 'currentModel.pkl')