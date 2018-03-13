import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import utils

from configs import myConfigs
import os
import pickle
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
            drop = F.dropout(x, training=True, p=self.p)
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

    start = time.time()

    gpu_id = 0
    config_number = 3
    config = myConfigs[config_number]

    model = MLP(784, config["hidden_layers"], 10, config["nonlinearity"], config["initialization"], config["dropout"], verbose=True)
    model.load_state_dict(torch.load(os.path.join("results", "config"+str(config_number), "model")))
    model.cuda(gpu_id)

    # Loading the MNIST dataset
    _, _, _, _, x_test, y_test = utils.load_mnist(config["data_file"], data_format=config["data_format"])

    x_test = Variable(torch.from_numpy(x_test), volatile=True).cuda(gpu_id)
    y_test = Variable(torch.from_numpy(y_test), volatile=True).cuda(gpu_id)

    def evaluate(data, labels):
        output = model(data, is_training=False)
        prediction = torch.max(output.data, 1)[1]
        accuracy = (prediction.eq(labels.data).sum() / labels.size(0)) * 100
        return accuracy

    test_acc_i = evaluate(x_test, y_test)
    print("Model Restored\nPrecision on test set : {}".format(test_acc_i))

    N_s = range(10, 101, 10)

    experiment_ii = []
    experiment_iii = []
    experiment_iv = []

    for N in N_s:

        print("\n----------\nN={}\n".format(N))

        # Experiment ii) - Pre-softmax ensemble
        pre_softmax_ensemble = Variable(torch.zeros(x_test.size()[0], 10, N), volatile=True).cuda(gpu_id)
        for j in range(N):
            pre_softmax_ensemble[:, :, j] = model.forward_until_pre_softmax(x_test, part=1)

        pre_softmax_avg = torch.mean(pre_softmax_ensemble, dim=2)
        output = model.forward_until_pre_softmax(pre_softmax_avg, part=2)
        prediction = torch.max(output.data, 1)[1]
        accuracy = (prediction.eq(y_test.data).sum() / y_test.size(0)) * 100
        experiment_ii.append(accuracy)

        # Experiment iii) - Post-softmax ensemble
        post_softmax_ensemble = Variable(torch.zeros(x_test.size()[0], 10, N), volatile=True).cuda(gpu_id)
        for j in range(N):
            post_softmax_ensemble[:, :, j] = model(x_test, is_training=True)

        post_softmax_avg = torch.mean(post_softmax_ensemble, dim=2)
        output = post_softmax_avg
        prediction = torch.max(output.data, 1)[1]
        accuracy = (prediction.eq(y_test.data).sum() / y_test.size(0)) * 100
        experiment_iii.append(accuracy)

        # Experiment iv) - Voting ensemble
        votes = Variable(torch.zeros(x_test.size()[0], 10), volatile=True).cuda(gpu_id)
        for j in range(N):
            output = model(x_test, is_training=True)

            prediction = torch.max(output.data, 1)[1]
            votes[range(10000), prediction] += 1

        final_prediction = torch.max(votes.data, 1)[1]
        accuracy = (final_prediction.eq(y_test.data).sum() / y_test.size(0)) * 100
        experiment_iv.append(accuracy)


    with open(os.path.join("results", "config"+str(config_number), "dropout_results.pkl"), 'wb') as f:
        pickle.dump({
            'N_s': N_s,
            'test_acc_i': test_acc_i,
            'experiment_ii': experiment_ii,
            'experiment_iii': experiment_iii,
            'experiment_iv': experiment_iv
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nTOTAL TIME : {}".format(time.time() - start))


