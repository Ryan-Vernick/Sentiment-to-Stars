from transformers import AutoModelForSequenceClassification, AutoTokenizer
from joblib import dump
from tokenization import tokenize

model = AutoModelForSequenceClassification.from_pretrained("model")
tokens = tokenize("Hello peace love")
print(model(**tokens))
dump(model, 'currentModel.pkl')