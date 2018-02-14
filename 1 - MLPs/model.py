import torch.nn as nn
import torch.nn.functional as F
import math

class MLP(nn.Module):

    def __init__(self, inp_size, h_sizes, out_size, act_fn, init_type, verbose=False):

        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList([nn.Linear(inp_size, h_sizes[0])])
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Activation function
        if act_fn == "relu":
            self.act_fn = F.relu

        elif act_fn == "sigmoid":
            self.act_fn = F.sigmoid

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

        # Initializes the parameters
        self.parameters_init(init_type)

        if verbose:
            print('\nModel Info ------------')
            print(self)
            print("Total number of parameters : ")
            print('---------------------- \n')

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            a = layer(x)
            x = self.act_fn(a)

        #output = self.out(x)
        output = F.log_softmax(self.out(x), dim=1)

        return output

    def parameters_init(self, init_type):

        for module in self.modules():
            if isinstance(module, nn.Linear):

                nn.init.constant(module.bias, 0)

                if init_type == "zero":
                    nn.init.constant(module.weight, 0)

                elif init_type == "normal":
                    nn.init.normal(module.weight, mean=0, std=1)

                elif init_type == "glorot":
                    nn.init.xavier_uniform(module.weight, gain=1)

                elif init_type == "default":
                    stdv = 1. / math.sqrt(module.weight.size(1))
                    nn.init.uniform(module.weight, -stdv, stdv)

        for p in self.parameters():
            p.requires_grad = True

    def name(self):
        return "MLP"
