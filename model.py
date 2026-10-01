import torch
import torch.nn as nn
import regressionhead
import config

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=3)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.head = regressionhead.RegressionHead(hidden_size)
    
    def forward(self, x):
        lengths = (x!=3).sum(dim=1)
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        _, hidden = self.rnn(packed, h0)
        # out = self.head(out[:, -1, :])
        out = self.head(hidden[-1])
        return out

vocab_size = config.vocab_size
embed_size = config.embed_size
hidden_size = config.hidden_size
output_size = config.output_size
model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)