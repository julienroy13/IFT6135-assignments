import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, inp_size, h_sizes, out_size):

        super(MLP, self).__init__()

        # Input layer
        self.inp = nn.Linear(inp_size, h_sizes[0])

        # Hidden layers
        self.hidden = []
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, x):

        # Feedforward
        x = F.relu(self.inp(x))
        for layer in self.hidden:
            x = F.relu(layer(x))
        pred = F.softmax(self.out(x), dim=1) # TODO: make sure this dim is right

        return pred

    def name(self):
        return "MLP"
