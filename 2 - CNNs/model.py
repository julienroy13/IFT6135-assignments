import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

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




class CNN(nn.Module):

    def __init__(self, inp_size, h_sizes, out_size, nonlinearity, init_type, is_batch_norm, verbose=False):

        super(CNN, self).__init__()

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn1 = nn.BatchNorm2d(16)

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        # Layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        # Logistic Regression
        self.fc = nn.Linear(128, 10)

        # Batch Norm on?
        self.batch_norm_on = is_batch_norm

        if verbose:
            print('\nModel Info ------------')
            print(self)
            print("Total number of parameters : {:.2f} M".format(self.get_number_of_params() / 1e6))
            print('---------------------- \n')

    def forward(self, x):
        # Layer 1
        out = F.relu(self.pool1(self.conv1(x)))
        if self.batch_norm_on:
            out = self.bn1(out)

        # Layer 2
        out = F.relu(self.pool2(self.conv2(out)))
        if self.batch_norm_on:
            out = self.bn2(out)

        # Layer 3
        out = F.relu(self.pool2(self.conv2(out)))
        if self.batch_norm_on:
            out = self.bn3(out)

        # Layer 4
        out = F.relu(self.pool2(self.conv2(out)))
        if self.batch_norm_on:
            out = self.bn4(out)

        # Final classifier
        out = F.log_softmax(self.fc(x), dim=1)
        
        return out

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
