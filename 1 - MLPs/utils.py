import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

def load_mnist(data_file):

    with open(data_file, "rb") as f:
        mnist = pickle.load(f)

    x_train = mnist['x_train']
    y_train = mnist['y_train']

    x_valid = mnist['x_valid']
    y_valid = mnist['y_valid']

    x_test = mnist['x_test']
    y_test = mnist['y_test']

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def reduce_trainset_size(x_train, y_train, reduction_coeff):

    rdm_state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(rdm_state)
    np.random.shuffle(y_train)

    limit = int(reduction_coeff * 50000)

    return x_train[:limit], y_train[:limit]


def save_model(model):

    pass # TODO
    return

def save_results(train_tape, valid_tape, test_tape, exp_name):

    plt.figure(figsize=(20, 6))

    n_epochs = len(valid_tape[0])
    epochs = np.arange(n_epochs)

    print(train_tape[1])
    print(valid_tape[1])
    print(test_tape[1])

    plt.subplot(1,2,1)
    plt.title("Loss")
    plt.plot(epochs, train_tape[0], color="violet", label="Training set")
    plt.plot(epochs, valid_tape[0], color="orange", label="Validation set")
    plt.plot(epochs, test_tape[0], color="blue", label="Test set")
    plt.xlabel("Epochs")
    plt.legend(loc='best')


    plt.subplot(1,2,2)
    plt.title("Accuracy")
    plt.plot(epochs, train_tape[1], color="violet", label="Training set")
    plt.plot(epochs, valid_tape[1], color="orange", label="Validation set")
    plt.plot(epochs, test_tape[1], color="blue", label="Test set")
    plt.ylim(0, 100)
    plt.xlabel("Epochs")
    plt.legend(loc='best')

    saving_dir = os.path.join("results", exp_name)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    plt.savefig(os.path.join(saving_dir, exp_name + '.png'))
    plt.close()

    return