import torch
import torch.nn as nn
import regressionhead
import config

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

vocab_size = config.vocab_size
embed_size = config.embed_size
hidden_size = config.hidden_size
output_size = config.output_size
model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)