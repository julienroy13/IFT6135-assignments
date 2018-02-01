import numpy as np
import pickle
import matplotlib.pyplot as plt

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

def save_results():

    pass # TODO
    return