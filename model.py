import torch
import torch.nn as nn
import regressionhead

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.head = regressionhead.RegressionHead(hidden_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.head(out[:, -1, :])
        return out

vocab_size = 50000 + 1 # !!!
embed_size = 128 # !!!
hidden_size = 128 # !!!
output_size = 2 # !!!
model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)