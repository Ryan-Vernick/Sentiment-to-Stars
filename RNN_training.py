import torch.nn as nn
import torch.optim as optim
import data_loader
import model
from data_loader import train_loader
from model import SentimentRNN, vocab_size, embed_size, hidden_size, output_size
import config

modell = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(modell.parameters(), lr=config.learning_rate)

batch_counter = 0
num_epochs = config.num_epochs
for epoch in range(num_epochs):
    modell.train()
    epoch_loss = 0
    for tokens, score in train_loader:
        outputs = modell(tokens)
        loss = criterion(outputs, score)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        print(f'Batch Loss: {loss.item():.4f}')
        batch_counter += 1
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
    print(f'Total Batches Processed: {batch_counter}')