from torch import tensor, long, float
from torch.utils.data import Dataset, DataLoader
from random import shuffle
from json import load
from config import training_set, testing_set, validation_set, batch_size

class SentimentDataset(Dataset):
    def __init__(self, data):
        self.tokens = [d['Tokens'] for d in data]
        self.score = [d['Score'] for d in data]

    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        score = self.score[idx]
        return tensor(tokens, dtype=long), tensor(score, dtype=float)
    
def load_data(train_set: list[int] | None = None, test_set: int | None = None, val_set: int | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Loads training, testing, and validattion data from the jsons

    Args:
        train_set (list[int] | None = None):
            The list of file numbers to load. If None, the default training files will be loaded
        test_set (int | None = None):
            The file number to load for testing. If None, the default testing file will be loaded
        validation_set (int | None = None):
            The file number to load for validation. If None, the default validation file will be loaded
        
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: The DataLoaders for the training, testing, and validation data
    """
    if train_set is None:
        train_set = training_set
    if test_set is None:
        test_set = testing_set
    if val_set is None:
        val_set = validation_set

    train_data = []
    for _,file_num in enumerate(train_set):
        with open(f'./vocab_jsons/vocab_{file_num}.json', 'r') as f:
            train_data.extend(load(f))
    #shuffle(train_data) # overkill but just to be sure

    with open(f'./vocab_jsons/vocab_{test_set}.json', 'r') as f:
        test_data = load(f)

    with open(f'./vocab_jsons/vocab_{val_set}.json', 'r') as f:
        validation_data = load(f)
    train_dataset = SentimentDataset(train_data)
    test_dataset = SentimentDataset(test_data)
    validation_dataset = SentimentDataset(validation_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, validation_loader