import torch
from torch.utils.data import Dataset, DataLoader
import random
import json
import config

with open('./vocab_jsons/vocab_1.json', 'r') as f: # !!!
    train_data = json.load(f)

random.shuffle(train_data) # overkill but just to be sure

def pad(text):
    return text + [0] * (2475 - len(text))

with open('./vocab_jsons/vocab_2.json', 'r') as f: # !!!
    test_data = json.load(f)

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.tokens = [d['Tokens'] for d in data]
        self.score = [d['Score'] for d in data]

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        score = self.score[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(score, dtype=torch.float)

for dict in train_data:
    dict['Tokens'] = pad(dict['Tokens'])

for dict in test_data:
    dict['Tokens'] = pad(dict['Tokens'])

train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)