import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import utils
from model import MLP

from configs import myConfigs
import os
import argparse

import numpy as np

torch.manual_seed(1234)

def train_model(config, config_number, gpu_id):

    # Instantiating the model
    model = MLP(784, config["hidden_layers"], 10)

    # Loading the MNIST dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_mnist(config["data_file"])

    # If GPU is available, sends model and dataset on the GPU
    if False and torch.cuda.is_available():
        model.cuda()

        x_train = torch.from_numpy(x_train).cuda(gpu_id)
        y_train = torch.from_numpy(y_train).cuda(gpu_id)

        x_valid = torch.from_numpy(x_valid).cuda(gpu_id)
        y_valid = torch.from_numpy(y_valid).cuda(gpu_id)

        x_test = torch.from_numpy(x_test).cuda(gpu_id)
        y_test = torch.from_numpy(y_test).cuda(gpu_id)
        print("Running on GPU")
    else:
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        x_valid = torch.from_numpy(x_valid)
        y_valid = torch.from_numpy(y_valid)

        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(y_test)
        print("WATCH-OUT : torch.cuda.is_available() returned False. Running on CPU.")

    # Instantiate TensorDataset and DataLoader objects
    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(train_set, batch_size=config["mb_size"], shuffle=True)

    # Optimizer and Loss Function
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    loss_fn = nn.CrossEntropyLoss()

    # TRAINING LOOP
    for epoch in range(config["max_epochs"]):
        for x_batch, y_batch in loader:

            # Empties the gradients
            optimizer.zero_grad()

            # Feedforward through the model
            pred = model(x_batch)

            # Computes the loss
            loss = loss_fn(pred, y_batch)

            # Backpropagates to compute the gradients
            loss.backward()

            # Takes one training step
            optimizer.step()

        # Evaluates the model on validation set
        pred = model(x_valid)
        loss = loss_fn(pred, y_valid)
        print("Epoch {0} : {1:.3f}".format(epoch, loss))

    if not os.path.exists("results"):
        os.makedirs(args.savedir)

    # Saves the graphs
    # TODO

    return








if __name__ == "__main__":
    # Retrieves arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True,
                        help='config id number')

    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id number')

    args = parser.parse_args()
    print(args)

    # Extracts the chosen config
    config_number = int(args.config)
    config = myConfigs[config_number]
    gpu_id = int(args.gpu)

    # Runs the training procedure
    print("Running the training procedure for config-{}".format(config_number))
    train_model(config, config_number, gpu_id)