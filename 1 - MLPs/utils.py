import numpy as np
import pickle

def load_mnist(data_file):

    with open("mnist_data.pkl", "rb") as f:
        mnist = pickle.load(f)

    x_train = mnist['x_train']
    y_train = mnist['y_train']

    x_valid = mnist['x_valid']
    y_valid = mnist['y_valid']

    x_test = mnist['x_test']
    y_test = mnist['y_test']

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def create_minibatches(train_data, train_labels, mb_size):
    """
    :param train_data: the whole tensor of training examples
    :param train_labels: the whole tensor of training labels
    :param mb_size: the size of each minibatch
    :return: a list of tuples containing the minibatches (train_data, train_labels)
    """
    rdm_state = np.random.get_state()
    np.random.shuffle(train_data)
    np.random.set_state(rdm_state)
    np.random.shuffle(train_labels)

    minibatches = []

    for i in range(int(np.ceil(train_data.shape[0] / mb_size))):

        minibatches.append((train_data[i*mb_size, i*mb_size + mb_size],
                            train_labels[i*mb_size, i*mb_size + mb_size]))

    return minibatches


def save_model(model):

    pass # TODO
    return

def save_results():

    pass # TODO
    return