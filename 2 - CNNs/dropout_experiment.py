import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import utils

from configs import myConfigs
import os
import argparse
import math
import time
import pdb

import numpy as np

torch.manual_seed(1234)


class MLP(nn.Module):

    def __init__(self, inp_size, h_sizes, out_size, nonlinearity, init_type, dropout, verbose=False):

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

        # Dropout
        self.p = dropout

        if verbose:
            print('\nModel Info ------------')
            print(self)
            print("Total number of parameters : {:.2f} M".format(self.get_number_of_params() / 1e6))
            print('---------------------- \n')

    def forward(self, x, is_training=False):

        # Feedforward
        for layer in self.hidden:
            a = layer(x)
            x = self.act_fn(a)

        # Dropout on last hidden layer
        drop = F.dropout(x, training=is_training, p=self.p)
        output = F.log_softmax(self.out(drop), dim=1)
        return output

    def forward_until_last_hidden(self, x, part):
        if part==1:
            # Feedforward
            for layer in self.hidden:
                a = layer(x)
                x = self.act_fn(a)

            # Dropout on last hidden layer
            drop = F.dropout(x, training=False, p=self.p)
            return drop

        elif part==2:
            # Finish forward propagation
            output = F.log_softmax(self.out(x), dim=1)
            return output

    def forward_until_pre_softmax(self, x, part):
        if part == 1:
            # Feedforward
            for layer in self.hidden:
                a = layer(x)
                x = self.act_fn(a)

            # Dropout on last hidden layer
            drop = F.dropout(x, training=False, p=self.p)
            pre_softmax = self.out(drop)
            return pre_softmax

        elif part == 2:
            # Finish forward propagation
            output = F.log_softmax(x, dim=1)
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

        weights_L2_norm_squared = 0

        for params in self.parameters():
            weights_L2_norm_squared += torch.sum(params**2)

        return torch.sqrt(weights_L2_norm_squared)

    def name(self):
        return "MLP"



if __name__ == "__main__":

    config_number = 3
    config = myConfigs[config_number]

    model = MLP(784, config["hidden_layers"], 10, config["nonlinearity"], config["initialization"], config["dropout"], verbose=True)
    model.load_state_dict(torch.load(os.path.join("results", "config"+str(config_number), "model")))

    # Loading the MNIST dataset
    _, _, _, _, x_test, y_test = utils.load_mnist(config["data_file"], data_format=config["data_format"])

    x_test = Variable(torch.from_numpy(x_test))
    y_test = Variable(torch.from_numpy(y_test))

    N = range(10, 110, 10)

