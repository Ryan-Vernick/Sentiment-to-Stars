from torch import zeros
from torch.nn import Module, Embedding, RNN
from torch.nn.utils.rnn import pack_padded_sequence
from regressionhead import RegressionHead
from config import vocab_size, embed_size, hidden_size, output_size, padding_id

class SentimentRNN(Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(SentimentRNN, self).__init__()
        self.embedding = Embedding(vocab_size, embed_size, padding_idx=padding_id)
        self.rnn = RNN(embed_size, hidden_size, batch_first=True)
        self.head = RegressionHead(hidden_size, output_size)
    def forward(self, x):
        device = x.device
        lengths = (x!=padding_id).sum(dim=1).cpu()
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h0 = zeros(1, x.size(0), hidden_size, device=device)
        _, hidden = self.rnn(packed, h0)
        # out = self.head(out[:, -1, :])
        x = self.head(hidden[-1])
        return x

if __name__ == "__main__":
    model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)
    print(model)