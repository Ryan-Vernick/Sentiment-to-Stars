import torch.nn as nn
import torch.optim as optim
import data_loader
import model
from data_loader import train_loader
from model import SentimentRNN, vocab_size, embed_size, hidden_size, output_size

modell = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modell.parameters(), lr=0.001) # !!!

num_epochs = 10 # !!!
for epoch in range(num_epochs):
    modell.train()
    epoch_loss = 0
    for texts, labels in train_loader:
        outputs = modell(texts)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')