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
        "hidden_layers": [512, 128],
        "activation": "relu", # "relu", "sigmoid"
        "initialization": "glorot", # "glorot", "zero", "normal"
        "mb_size": 1000,
        "max_epochs": 10,

        "lr": 0.01,
        "momentum": 0.9,

        "save_plots":True
        }
     ),

]))