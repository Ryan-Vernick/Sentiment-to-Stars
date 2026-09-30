import torch
from torch.utils.data import Dataset, DataLoader
import random
import json

with open('./vocab_jsons/vocab_1.json', 'r') as f: # !!!
    train_data = json.load(f)

random.shuffle(train_data) # overkill but just to be sure

with open('./vocab_jsons/vocab_2.json', 'r') as f: # !!!
    test_data = json.load(f)

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.tokens = data['Tokens'].values
        self.score = data['Score'].values
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        score = self.score[idx]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(score, dtype=torch.float)

train_dataset = SentimentDataset(train_data)
test_dataset = SentimentDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) # !!!
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) # !!!