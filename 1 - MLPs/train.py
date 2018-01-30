import torch
import torch.optim as optim

import utils
from model import MLP

from configs import myConfigs
import os
import argparse

import numpy as np
np.random.seed(1234)

def train_model(config, config_number):

    # Loading the MNIST dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_mnist(config["data_file"])

    for epoch in range(config("max_epochs")):
        minibatches = utils.create_minibatches(x_train, y_train, config["mb_size"])

        for x_minibatch, y_minibatch in minibatches:

            pass
            # TODO : Training loop






    if not os.path.exists("results"):
        os.makedirs(args.savedir)

    # Saves the graphs

    return








if __name__ == "__main__":
    # Retrieves arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=int, required=True,
                        choices=['mnist', '20documents'],
                        help='config id number')

    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id number')

    args = parser.parse_args()
    print(args)

    # Extracts the chosen config
    config_number = args.config
    config = myConfigs[config_number]

    # Runs the training procedure
    print("Running the training procedure for config-{}".format(config_number))
    train_model(config, config_number)


