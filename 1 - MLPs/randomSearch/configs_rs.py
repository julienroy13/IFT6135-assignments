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


rsConfigs = (CarefulDict([
    
    (0, {
        "hp_to_search": ["activation", "initialization", "mb_size", "lr"],
        "n_samples": 5,

        "data_file": os.path.join("data", "mnist_data.pkl"),
        "data_reduction": 1.0, # 0.01, 0.02, 0.05, 0.1, 1.0
        
        "hidden_layers": [1024, 512],
        "activation": ("ind", ["relu", "sigmoid"]),
        "initialization": ("ind", ["default", "glorot", "zero", "normal"]),
        
        "mb_size": ("int", [10, 1000]),
        "max_epochs": 2,

        "lr": ("exp", [-5, 0]),
        "momentum": 0.9,

        "show_test": False,
        "save_plots": True
        }
    ),

]))