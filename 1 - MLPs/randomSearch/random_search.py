import sys
sys.path.append('..')

import train

import numpy as np

from termcolor import cprint
import argparse
import traceback
import math
import os
import copy
from configs_rs import rsConfigs

def write_description_file(config, rs_config, save_dir, exp_name):

    with open(os.path.join(save_dir, exp_name, 'description.txt'), 'w') as f:
        f.write("RANDOM SEARCH CONFIGS") 
        for h in rs_config.keys():
            f.write(h+" : "+str(rs_config[h])+"\n")

        f.write("\n\nHYPERPARAMS SPECIFIC TO THIS SAMPLE\n") 
        for h in rs_config["hp_to_search"]:
            f.write(h+" : "+str(config[h])+"\n")


if __name__ == "__main__":
    # Retrieves arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--rsconfig', type=str, default='0',
                        help='config_rs id number')

    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id number')

    args = parser.parse_args()
    print(args)

    # Extracts the chosen config
    rs_config_number = int(args.rsconfig)
    rs_config = rsConfigs[rs_config_number]
    gpu_id = int(args.gpu)
    save_dir = os.path.join("results", "search"+str(rs_config_number))

    for i in range(rs_config["n_samples"]):

        config = copy.deepcopy(rs_config)
        for hyperparam in rs_config["hp_to_search"]:

            if rs_config[hyperparam][0] == "ind":
                index = np.random.randint(low=0, high=len(rs_config[hyperparam][1]))
                config[hyperparam] = rs_config[hyperparam][1][index]

            elif rs_config[hyperparam][0] == "int":
                value = np.random.randint(low=rs_config[hyperparam][1][0], high=rs_config[hyperparam][1][1] + 1)
                config[hyperparam] = value

            elif rs_config[hyperparam][0] == "exp":
                exp = np.random.randint(low=rs_config[hyperparam][1][0], high=rs_config[hyperparam][1][1] + 1)
                config[hyperparam] =math.pow(10, exp)

        exp_name = "sample" + str(i+1)

        try:
            # Runs the training procedure
            cprint("Running the training procedure for sample-{}".format(i), "blue")

            if not os.path.exists(os.path.join(save_dir, exp_name)):
                os.makedirs(os.path.join(save_dir, exp_name))

            write_description_file(config, rs_config, save_dir, exp_name)
            train.train_model(config, gpu_id, save_dir, exp_name)

        except Exception as e: 
            cprint("Error occured in experiment for sample {} \n----".format(i), "red")
            traceback.print_exc()
            cprint("----", "red")
        print("\n")