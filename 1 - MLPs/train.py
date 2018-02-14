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
import math
import time
import pdb

import numpy as np

torch.manual_seed(1234)


def train_model(config, gpu_id, save_dir, exp_name):
    # Instantiating the model
    model = MLP(784, config["hidden_layers"], 10, config["activation"], config["initialization"], verbose=True)

    # Loading the MNIST dataset
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_mnist(config["data_file"])

    if config['data_reduction'] != 1.:
        x_train, y_train = utils.reduce_trainset_size(x_train, y_train, config['data_reduction'])

    # If GPU is available, sends model and dataset on the GPU
    if torch.cuda.is_available():
        model.cuda(gpu_id)

        x_train = torch.from_numpy(x_train).cuda(gpu_id)
        y_train = torch.from_numpy(y_train).cuda(gpu_id)

        x_valid = Variable(torch.from_numpy(x_valid)).cuda(gpu_id)
        y_valid = Variable(torch.from_numpy(y_valid)).cuda(gpu_id)

        x_test = Variable(torch.from_numpy(x_test)).cuda(gpu_id)
        y_test = Variable(torch.from_numpy(y_test)).cuda(gpu_id)
        print("Running on GPU")
    else:
        x_train = torch.from_numpy(x_train)
        y_train = torch.from_numpy(y_train)

        x_valid = Variable(torch.from_numpy(x_valid))
        y_valid = Variable(torch.from_numpy(y_valid))

        x_test = Variable(torch.from_numpy(x_test))
        y_test = Variable(torch.from_numpy(y_test))
        print("WATCH-OUT : torch.cuda.is_available() returned False. Running on CPU.")

    # Instantiate TensorDataset and DataLoader objects
    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    loader = torch.utils.data.DataLoader(train_set, batch_size=config["mb_size"], shuffle=True)

    # Optimizer and Loss Function
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    loss_fn = nn.NLLLoss()
    #loss_fn = nn.CrossEntropyLoss()

    # Records the model's performance
    train_tape = [[],[]]
    valid_tape = [[],[]]
    test_tape = [[],[]]

    def evaluate(data, labels):

        if not isinstance(data, Variable):
            if torch.cuda.is_available():
                data = Variable(data).cuda(gpu_id)
                labels = Variable(labels).cuda(gpu_id)
            else:
                data = Variable(data)
                labels = Variable(labels)

        output = model(data)
        loss = loss_fn(output, labels)
        prediction = torch.max(output.data, 1)[1]
        accuracy = (prediction.eq(labels.data).sum() / labels.size(0)) * 100

        return loss.data[0], accuracy

    # Record train accuracy
    train_loss, train_acc = evaluate(x_train, y_train)
    train_tape[0].append(train_loss)
    train_tape[1].append(train_acc)

    # Record valid accuracy
    valid_loss, valid_acc = evaluate(x_valid, y_valid)
    valid_tape[0].append(valid_loss)
    valid_tape[1].append(valid_acc)

    # Record test accuracy
    test_loss, test_acc = evaluate(x_test, y_test)
    test_tape[0].append(test_loss)
    test_tape[1].append(test_acc)

    print("BEFORE TRAINING \nLoss : {0:.3f} \nAcc : {1:.3f}".format(valid_loss, valid_acc))

    # TRAINING LOOP
    for epoch in range(1, config["max_epochs"]):
        start = time.time()
        for i,(x_batch, y_batch) in enumerate(loader):

            #pdb.set_trace()

            if torch.cuda.is_available():
                x_batch = Variable(x_batch).cuda(gpu_id)
                y_batch = Variable(y_batch).cuda(gpu_id)
            else:
                x_batch = Variable(x_batch)
                y_batch = Variable(y_batch)

            # Empties the gradients
            optimizer.zero_grad()

            # Feedforward through the model
            output = model(x_batch)

            # Computes the loss
            loss = loss_fn(output, y_batch)
            #print(i, loss)

            # Backpropagates to compute the gradients
            loss.backward()

            # Takes one training step
            optimizer.step()

        # Record train accuracy
        train_loss, train_acc = evaluate(x_train, y_train)
        train_tape[0].append(train_loss)
        train_tape[1].append(train_acc)

        # Record valid accuracy
        valid_loss, valid_acc = evaluate(x_valid, y_valid)
        valid_tape[0].append(valid_loss)
        valid_tape[1].append(valid_acc)

        # Record test accuracy
        test_loss, test_acc = evaluate(x_test, y_test)
        test_tape[0].append(test_loss)
        test_tape[1].append(test_acc)

        print("Epoch {0} \nLoss : {1:.3f} \nAcc : {2:.3f}".format(epoch, valid_loss, valid_acc))
        print("Time : {0:.2f}".format(time.time() - start))



    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Saves the graphs
    utils.save_results(train_tape, valid_tape, test_tape, save_dir, exp_name, config['data_file'], config['show_test'])
    utils.update_comparative_chart(save_dir, config['show_test'])

    return



if __name__ == "__main__":
    # Retrieves arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='0',
                        help='config id number')

    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id number')

    args = parser.parse_args()
    print(args)

    # Extracts the chosen config
    config_number = int(args.config)
    config = myConfigs[config_number]
    gpu_id = int(args.gpu)
    save_dir = "results"
    exp_name = "config" + str(config_number)

    # Runs the training procedure
    print("Running the training procedure for config-{}".format(config_number))
    train_model(config, gpu_id, save_dir, exp_name)