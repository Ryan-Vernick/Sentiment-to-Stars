from torch import long, no_grad, device
from torch.cuda import is_available as cuda_is_available
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import load_data
from model import SentimentRNN, vocab_size, embed_size, hidden_size, output_size
from config import num_epochs

def get_device() -> device:
    """
    Returns the device to use for training and testing

    Returns:
        torch.device: The device to use for training and testing
    """
    device_type = 'cuda' if cuda_is_available() else 'cpu'
    return device(device_type)

def train_model(model: SentimentRNN | None = None, train_loader: DataLoader | None = None, output: bool = False) -> SentimentRNN:
    """
    Trains the model on the training data

    Args:
        model (SentimentRNN | None):
            The model to train. If None, a new model will be created
        train_loader (DataLoader | None):
            The DataLoader for the training data. If None, the default data will be loaded
        output (bool = False):
            Whether to print the training progress

    Returns:
        SentimentRNN: The trained model
    """
    #initialization
    device = get_device()
    if model is None:
        model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size).to(device)

    if train_loader is None:
        train_loader = load_data()[0]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_counter = 0
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for tokens, score in train_loader:
            tokens = tokens.to(device)
            score = (score - 1).to(device, dtype=long)
            optimizer.zero_grad()
            outputs = model(tokens)
            outputs = ((outputs*5).trunc())/5.0
            loss = criterion(outputs, score)
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if output and batch_counter % 100 == 0:
                pred = outputs.argmax(dim=1)
                acc = (pred == score).float().mean()
        
                print(
                    f"{batch_counter:4d}  "
                    f"loss={loss.item():.4f}  "
                    f"acc={acc.item():.3f}"
                )
            batch_counter += 1
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
        print(f'Total Batches Processed: {batch_counter}')
    
    return model

def test_model(model: SentimentRNN, test_loader: DataLoader | None = None, output: bool = False) -> None:
    """
    Tests the model on the test data

    Args:
        model (SentimentRNN):
            The model to test
        test_loader (DataLoader | None):
            The DataLoader for the test data. If None, the default data will be loaded
        output (bool = False):
            Whether to print the intermediate test progress
    """
    #initialization
    device = get_device()
    model.to(device)
    if test_loader is None:
        test_loader = load_data()[1]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_counter = 0
    total_loss = 0
    total_acc = 0
    #test loop
    with no_grad():
        for tokens, score in test_loader:
            tokens = tokens.to(device)
            score = (score - 1).to(long).to(device)
            outputs = model(tokens)
            outputs = ((outputs*5).trunc())/5.0
            loss = criterion(outputs, score)
            pred = outputs.argmax(dim=1)
            acc = (pred == score).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            batch_counter += 1
            if output and batch_counter % 100 == 0:
                print(
                    f"Test Loss: {loss.item():.4f}  "
                    f"Test Acc: {acc.item():.3f}"
                )
    print(f"Average Test Loss: {total_loss / batch_counter:.4f}")
    print(f"Average Test Accuracy: {total_acc / batch_counter:.3f}")

def validate_model(model: SentimentRNN, validation_loader: DataLoader | None = None, output: bool = False) -> None:
    """
    Validates the model on the validation data

    Args:
        model (SentimentRNN):
            The model to validate
        validation_loader (DataLoader | None):
            The DataLoader for the validation data. If None, the default data will be loaded
        output (bool = False):
            Whether to print the intermediate validation progress
    """
    #initialization
    device = get_device()
    model.to(device)
    if validation_loader is None:
        validation_loader = load_data()[2]
    criterion = nn.CrossEntropyLoss()
    model.eval()
    batch_counter = 0
    total_loss = 0
    total_acc = 0
    #validate loop
    with no_grad():
        for tokens, score in validation_loader:
            tokens = tokens.to(device)
            score = (score - 1).to(long).to(device)
            outputs = model(tokens)
            outputs = ((outputs*5).trunc())/5.0
            loss = criterion(outputs, score)
            pred = outputs.argmax(dim=1)
            acc = (pred == score).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()
            batch_counter += 1
            if output and batch_counter % 100 == 0:
                print(f"Validation Loss: {loss.item():.4f}\nValidation Acc: {acc.item():.3f}")
    #final output
    from config import learning_rate
    print(f"Average Validation Loss: {total_loss / batch_counter:.4f}")
    print(f"Average Validation Accuracy: {total_acc / batch_counter:.3f}")
    print("Hyper parameters:")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {validation_loader.batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Vocabulary Size: {vocab_size}")
    print(f"Embedding Size: {embed_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Loss Function: {type(criterion)}")
    print("Optimizer: Adam")
    print(f"Model Architecture: {model}")

if __name__ == "__main__":
    train, test, validation = load_data()
    model = train_model(train_loader=train, output=True)
    validate_model(model, validation_loader=validation, output=True)
    test_model(model, test_loader=test, output=True)
    