import torch
import torch.nn as nn
import torch.optim as optim
import data_loader
import model
from data_loader import train_loader, test_loader
from model import SentimentRNN, vocab_size, embed_size, hidden_size, output_size
import config

modell = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modell.parameters(), lr=1e-3)
batch_counter = 0
num_epochs = config.num_epochs
modell.train()
for epoch in range(10):
    epoch_loss = 0
    for tokens, score in train_loader:
        score = (score - 1).to(torch.long)  # Convert to long tensor for CrossEntropyLoss
        optimizer.zero_grad()
        outputs = modell(tokens)
        outputs = ((outputs*5).trunc())/5.0
        loss = criterion(outputs, score)
        loss.backward()
        #nn.utils.clip_grad_norm_(modell.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        if batch_counter % 100 == 0:
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

# tokens, labels = next(iter(train_loader))
# labels = (labels - 1).to(torch.long)
# modell.train()
# for epoch in range(1200):
#     optimizer.zero_grad()
#     output = modell(tokens)
#     loss = criterion(output, labels)
#     # pred = output.argmax(dim=1)

#     # accuracy = (pred == labels).float().mean()
#     # print("output:", output[:5].detach())
#     # print("target:", labels[:5].detach())
#     # print("loss:", loss.item())
#     # print("accuracy:", accuracy.item())

#     loss.backward()
#     #nn.utils.clip_grad_norm_(modell.parameters(), max_norm=1.0)
#     optimizer.step()
#     if epoch % 100 == 0:
#         pred = output.argmax(dim=1)
#         acc = (pred == labels).float().mean()

#         print(
#             f"{epoch:4d}  "
#             f"loss={loss.item():.4f}  "
#             f"acc={acc.item():.3f}"
#         )

# with torch.no_grad():
#     logits = modell(tokens)
#     probs = torch.softmax(logits, dim=1)
#     predictions = logits.argmax(dim=1)

# print("labels:     ", labels)
# print("predictions:", predictions)
# print("probs:\n", probs)

modell.eval()
for tokens, score in test_loader:
    score = (score - 1).to(torch.long)
    outputs = modell(tokens)
    outputs = ((outputs*5).trunc())/5.0
    loss = criterion(outputs, score)
    pred = outputs.argmax(dim=1)
    acc = (pred == score).float().mean()
    
    print(
        f"Test Loss: {loss.item():.4f}  "
        f"Test Acc: {acc.item():.3f}"
    )