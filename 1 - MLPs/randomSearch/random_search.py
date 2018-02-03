import sys
sys.path.append('..')

import train

import numpy as np

import argparse
import math
import os
import copy
from configs_rs import rsConfigs

if __name__ == "__main__":
    # Retrieves arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='0',
                        help='config_rs id number')

    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id number')

    args = parser.parse_args()
    print(args)

    # Extracts the chosen config
    config_number = int(args.config)
    rs_config = rsConfigs[config_number]
    gpu_id = int(args.gpu)
    save_dir = os.path.join("results", "search"+str(config_number))

    for i in range(rs_config["n_samples"]):

        config = copy.deepcopy(rs_config)
        for hyperparam in rs_config["hp_to_search"]:

            if rs_config["hyperparam"][0] == "ind":
                index = np.random.randint(low=0, high=len(rs_config["hyperparam"][1]))
                config[hyperparam] = rs_config["hyperparam"][1][index]

            elif rs_config["hyperparam"][0] == "int":
                value = np.random.randint(low=rs_config["hyperparam"][1][0], high=rs_config["hyperparam"][1][1] + 1)
                config[hyperparam] = value

            elif rs_config["hyperparam"][0] == "exp":
                exp = np.random.randint(low=rs_config["hyperparam"][1][0], high=rs_config["hyperparam"][1][1] + 1)
                config[hyperparam] =math.pow(10, exp)

        exp_name = "sample" + str(i+1)

        # Runs the training procedure
        print("Running the training procedure for sample-{}".format(i))
        train.train_model(config, gpu_id, save_dir, exp_name)