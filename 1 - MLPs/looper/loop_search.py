import sys
sys.path.append('..')

import train
import utils

import numpy as np

import pickle
from termcolor import cprint
import argparse
import traceback
import os
import copy
from configs_loop import loopConfigs



def create_result_text_file(save_dir, out_filename, hyperparam_studied):

    # Logs from all the experiments in that directory (each experiment has its own directory)
    all_logs = utils.get_all_logs_files(save_dir)

    # All values that have been looped over for the studied hyperparameter
    all_possible_values = sorted(set([log.config[hyperparam_studied] for log in all_logs]))

    # Dict that will contain several sample of generalization gap for every tested values of hyperparameter studied
    stats = {}
    for value in all_possible_values:
        line = hyperparam_studied + "=" + str(value)
        stats[line] = []

    # Go gets the stat of interest (generalization gap) in every log and store it in stats
    for log in all_logs:

        # Computes the generalization gap at best validation epoch
        best_epoch = np.argmax(log.valid_tape[1])
        gap = log.train_tape[1][best_epoch] - log.test_tape[1][best_epoch]

        # Stores it in stats
        line = hyperparam_studied + "=" + str(log.config[hyperparam_studied])
        stats[line].append(gap)

    stds = {}
    means = {}
    for value in all_possible_values:
        line = hyperparam_studied + "=" + str(value)

        means[line] = np.mean(stats[line])
        stds[line] = np.std(stats[line])

    # Writes those results in a text file
    with open(os.path.join(save_dir, out_filename), 'w') as f:

        for line in stats.keys():
            f.write(line + '\t')
            f.write(str(np.round(stats[line], 3)) + "\t")
            f.write('mean={0:.3f}\t'.format(means[line]))
            f.write('std={0:.3f}\t'.format(stds[line]))
            f.write('\n')
    return


if __name__ == "__main__":
    # Retrieves arguments from the command line
    parser = argparse.ArgumentParser()

    parser.add_argument('--loopconfig', type=str, default='0',
                        help='config_rs id number')

    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu id number')

    args = parser.parse_args()
    print(args)

    # Extracts the chosen config
    loop_config_number = int(args.loopconfig)
    loop_config = loopConfigs[loop_config_number]

    hyperparam_studied = loop_config["hyperparam_studied"]

    gpu_id = int(args.gpu)
    save_dir = os.path.join("results", "search"+str(loop_config_number))

    for v_id, value in enumerate(loop_config[hyperparam_studied]):

        config = copy.deepcopy(loop_config)
        config[hyperparam_studied] = value
        for i in range(loop_config["n_samples"]):

            exp_name = hyperparam_studied + str(v_id) + "_run" + str(i+1)

            try:
                # Runs the training procedure
                cprint("Running the training procedure for sample-{}".format(i), "blue")

                if not os.path.exists(os.path.join(save_dir, exp_name)):
                    os.makedirs(os.path.join(save_dir, exp_name))

                train.train_model(config, gpu_id, save_dir, exp_name)

            except Exception as e:
                cprint("Error occured in experiment for sample {} \n----".format(i), "red")
                traceback.print_exc()
                cprint("----", "red")
            print("\n")

    create_result_text_file(save_dir, "stats.txt", hyperparam_studied)