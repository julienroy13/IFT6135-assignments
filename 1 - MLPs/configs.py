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
    
    (0, {
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_reduction": 1.0, # 0.01, 0.02, 0.05, 0.1, 1.0
        
        "hidden_layers": [512, 512],
        "activation": "relu", # "relu", "sigmoid"
        "initialization": "default", # "default", "glorot", "zero", "normal"
        
        "mb_size": 100,
        "max_epochs": 100,

        "lr": 0.01,
        "momentum": 0.0,

        "show_test": False,
        "save_plots": True
        }
    ),

    (1, {
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "hidden_layers": [512, 512],
        "activation": "relu",  # "relu", "sigmoid"
        "initialization": "glorot",  # "default", "glorot", "zero", "normal"

        "mb_size": 100,
        "max_epochs": 100,

        "lr": 0.01,
        "momentum": 0.0,

        "show_test": False,
        "save_plots": True
        }
    ),

    (2, {
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "hidden_layers": [512, 512],
        "activation": "relu",  # "relu", "sigmoid"
        "initialization": "normal",  # "default", "glorot", "zero", "normal"

        "mb_size": 100,
        "max_epochs": 100,

        "lr": 0.01,
        "momentum": 0.0,

        "show_test": False,
        "save_plots": True
        }
    ),

    (3, {
        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_reduction": 1.0,  # 0.01, 0.02, 0.05, 0.1, 1.0

        "hidden_layers": [512, 512],
        "activation": "relu",  # "relu", "sigmoid"
        "initialization": "zero",  # "default", "glorot", "zero", "normal"

        "mb_size": 100,
        "max_epochs": 100,

        "lr": 0.01,
        "momentum": 0.0,

        "show_test": False,
        "save_plots": True
    }
     ),

]))