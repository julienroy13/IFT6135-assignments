import collections
import os

class CarefulDict(dict):
    def __init__(self, inp=None):
        # Checks is the input to constructor is already a dict
        if isinstance(inp,dict):
            # If yes, just initializes as usual
            super(CarefulDict,self).__init__(inp)
        else:
            # If not, set each item individually with the overriden __setitem__ method
            super(CarefulDict,self).__init__()
            if isinstance(inp, (collections.Mapping, collections.Iterable)): 
                for k,v in inp:
                    self.__setitem__(k,v)

    def __setitem__(self, k, v):
        try:
            # Looks if the key already exists, if it does, raises an error
            self.__getitem__(k)
            raise ValueError('duplicate key "{0}" found'.format(k))
        
        except KeyError:
            # If the key is unique, just add the tuple to the dict
            super(CarefulDict, self).__setitem__(k, v)


myConfigs = (CarefulDict([

    (0, { # Optimal config found by random-serach for [784, 512, 512, 10] MLP
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_format": "vector", # "vector" or "array"
        "data_reduction": 1.0, # 0.01, 0.02, 0.05, 0.1, 1.0
        
        "hidden_layers": [512, 512],
        "nonlinearity": "sigmoid", # "relu", "sigmoid", "tanh"
        "initialization": "glorot", # "standard", "glorot", "zero", "normal"
        
        "mb_size": 40,
        "max_epochs": 100,
        "patience": 5, # For early stopping, if no early stopping, set patience to None TODO

        "lr": 1.0,
        "momentum": 0.5,

        "is_early_stopping" : False, #True or False
        "L2_hyperparam" : 0, # L2 hyperparameter for a full batch (entire dataset)

        "show_test": False,
        "save_plots": True
        }
    ),

    (1, { # Imposed hyperparams for Assignment 2, Q1-a (WITH L2-REG)
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_format": "vector",  # "vector" or "array"
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "hidden_layers": [800, 800],
        "nonlinearity": "relu",  # "relu", "sigmoid", "tanh"
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 64,
        "max_epochs": 100,

        "lr": 0.02,
        "momentum": 0.0,
        "dropout": 0.0, # put 0 for no dropout

        "is_early_stopping": False,  # True or False
        "L2_hyperparam": 2.5,  # L2 hyperparameter for a full batch (entire dataset)

        "show_test": False,
        "save_plots": True
    }
     ),

    (2, { # Imposed hyperparams for Assignment 2, Q1-a (NO L2-REG)
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_format": "vector",  # "vector" or "array"
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "hidden_layers": [800, 800],
        "nonlinearity": "relu",  # "relu", "sigmoid", "tanh"
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 64,
        "max_epochs": 100,

        "lr": 0.02,
        "momentum": 0.0,
        "dropout": 0.0, # put 0 for no dropout

        "is_early_stopping": False,  # True or False
        "L2_hyperparam": 0,  # L2 hyperparameter for a full batch (entire dataset)

        "show_test": False,
        "save_plots": True
    }
     ),

    (3, { # Imposed hyperparams for Assignment 2, Q1-b (Dropout)
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_format": "vector",  # "vector" or "array"
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "hidden_layers": [800, 800],
        "nonlinearity": "relu",  # "relu", "sigmoid", "tanh"
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 64,
        "max_epochs": 100,

        "lr": 0.02,
        "momentum": 0.0,
        "dropout": 0.5, # put 0 for no dropout

        "is_early_stopping": False,  # True or False
        "L2_hyperparam": 0,  # L2 hyperparameter for a full batch (entire dataset)

        "show_test": True,
        "save_plots": True
    }
     ),

    (4, {  # Imposed hyperparams for Assignment 2, Q1-c (CNN and BatchNorm)
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_format": "array",  # "vector" or "array"
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "model_type": "CNN", # CNN or MLP
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 64,
        "max_epochs": 100,

        "lr": 0.02,
        "momentum": 0.0,
        "is_batch_norm": True,

        "is_early_stopping": False,  # True or False
        "L2_hyperparam": 0,  # L2 hyperparameter for a full batch (entire dataset)

        "show_test": True,
        "save_plots": True
    }
     ),

    (5, {  # Imposed hyperparams for Assignment 2, Q1-c (CNN without BatchNorm)
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_format": "array",  # "vector" or "array"
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "model_type": "CNN",  # CNN or MLP
        "initialization": "glorot",  # "standard", "glorot", "zero", "normal"

        "mb_size": 64,
        "max_epochs": 100,

        "lr": 0.02,
        "momentum": 0.0,
        "is_batch_norm": False,

        "is_early_stopping": False,  # True or False
        "L2_hyperparam": 0,  # L2 hyperparameter for a full batch (entire dataset)

        "show_test": True,
        "save_plots": True
    }
     ),



]))