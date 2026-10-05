import torch
import torch.nn as nn

# usign relu activation function and sigmoid for output to be in range [0, 1]
class RegressionHead(nn.Module):

    def __init__(self, hidden_dim: int):
        super(RegressionHead, self).__init__()

        #Layers of the onion, using // since i dont know the size of the hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc3 = nn.Linear(hidden_dim // 4, 5)

        self.relu = nn.ReLU()

        # output function 0-1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts: x of shape (batch_size, hidden_dim)

        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))

        # (batch_size, 5), values in [0, 1]
        output = self.fc3(h)
        return output