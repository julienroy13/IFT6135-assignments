import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

class MLP(nn.Module):

    def __init__(self, inp_size, h_sizes, out_size, nonlinearity, init_type, verbose=False):

        super(MLP, self).__init__()

        # Hidden layers
        self.hidden = nn.ModuleList([nn.Linear(inp_size, h_sizes[0])])
        for k in range(len(h_sizes)-1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))

        # Activation function
        self.nonlenarity = nonlinearity
        if nonlinearity == "relu":
            self.act_fn = F.relu

        elif nonlinearity == "sigmoid":
            self.act_fn = F.sigmoid

        elif nonlinearity == "tanh":
            self.act_fn = F.tanh

        else:
            raise ValueError('Specified activation function "{}" is not recognized.'.format(nonlinearity))

        # Output layer
        self.out = nn.Linear(h_sizes[-1], out_size)

        # Initializes the parameters
        self.init_parameters(init_type)

        if verbose:
            print('\nModel Info ------------')
            print(self)
            print("Total number of parameters : {:.2f} M".format(self.get_number_of_params() / 1e6))
            print('---------------------- \n')

    def forward(self, x):

        # Feedforward
        for layer in self.hidden:
            a = layer(x)
            x = self.act_fn(a)

        output = F.log_softmax(self.out(x), dim=1)

        return output

    def init_parameters(self, init_type):

        for module in self.modules():
            if isinstance(module, nn.Linear):

                nn.init.constant(module.bias, 0)

                if init_type == "glorot":
                    nn.init.xavier_normal(module.weight, gain=nn.init.calculate_gain(self.nonlenarity))

                elif init_type == "standard":
                    stdv = 1. / math.sqrt(module.weight.size(1))
                    nn.init.uniform(module.weight, -stdv, stdv)

        for p in self.parameters():
            p.requires_grad = True

    def get_number_of_params(self):

        total_params = 0

        for params in self.parameters():

            total_size = 1
            for size in params.size():
                total_size *= size

            total_params += total_size

        return total_params

    def get_weights_L2_norm(self):

        weights_L2_norm = 0

        pdb.set_trace()
        for params in self.parameters():
            weights_L2_norm += params.norm(2)

        return weights_L2_norm

    def name(self):
        return "MLP"




class CNN(nn.Module):

    def __init__(self, inp_size, h_sizes, out_size, nonlinearity, init_type, verbose=False):

        super(MLP, self).__init__()

        #TODO

        if verbose:
            print('\nModel Info ------------')
            print(self)
            print("Total number of parameters : {:.2f} M".format(self.get_number_of_params() / 1e6))
            print('---------------------- \n')

    def forward(self, x):

        # Feedforward
        #TODO
        pass

        return

    def init_parameters(self, init_type):

        for module in self.modules():
            if isinstance(module, nn.Linear):

                nn.init.constant(module.bias, 0)

                if init_type == "glorot":
                    nn.init.xavier_normal(module.weight, gain=nn.calculate_gain(self.nonlenarity))

                elif init_type == "standard":
                    stdv = 1. / math.sqrt(module.weight.size(1))
                    nn.init.uniform(module.weight, -stdv, stdv)

        for p in self.parameters():
            p.requires_grad = True

    def get_number_of_params(self):

        total_params = 0

        for params in self.parameters():

            total_size = 1
            for size in params.size():
                total_size *= size

            total_params += total_size

        return total_params

    def name(self):
        return "CNN"
