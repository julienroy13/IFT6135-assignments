import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import operator

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

def save_results(train_tape, valid_tape, test_tape, exp_name, data_file):

    # Creates the folder if necessary
    saving_dir = os.path.join("results", exp_name)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    # Creates and save the plots
    plt.figure(figsize=(20, 6))

    n_epochs = len(valid_tape[0])
    epochs = np.arange(n_epochs)

    plt.subplot(1,2,1)
    plt.title("Loss", fontweight='bold')
    plt.plot(epochs, train_tape[0], color="blue", label="Training set")
    plt.plot(epochs, valid_tape[0], color="orange", label="Validation set")
    plt.plot(epochs, test_tape[0], color="purple", label="Test set")
    plt.xlabel("Epochs")
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.title("Accuracy", fontweight='bold')
    plt.plot(epochs, train_tape[1], color="blue", label="Training set")
    plt.plot(epochs, valid_tape[1], color="orange", label="Validation set")
    plt.plot(epochs, test_tape[1], color="purple", label="Test set")
    plt.ylim(0, 100)
    plt.xlabel("Epochs")
    plt.legend(loc='best')

    plt.savefig(os.path.join(saving_dir, exp_name + '.png'))
    plt.close()

    # Save the recording tapes (learning curves) in a file
    log_file = os.path.join(saving_dir, 'log_' + exp_name + '.pkl')
    with open(log_file, 'wb') as f:
        pkl.dump({
            'data_file': data_file,
            'train_tape': train_tape,
            'valid_tape': valid_tape,
            'test_tape': test_tape
            })

    return


def init_comparative_chart():
    
    all_configs_results = []

    # Finds the right folders
    result_folders = os.listdir("results")
    for folder in result_folders:
        
        if folder.startswith('config'):
            config_number = int(folder.lstrip('config'))
            log_file      = os.path.join('results', folder, 'log_config'+config_number+'.pkl')

            # If the log_file exists, extracts the recording tapes it contains
            if os.path.exists(log_file):
                
                with open(log_file, 'rb') as f:
                    log_data = pkl.load(f)
                
                data_file = log_data['data_file']
                
                train_tape = log_data['train_tape']
                valid_tape = out_data['valid_tape']
                test_tape  = out_data['test_tape']
                
                # If tape is not empty, collects the results
                if len(valid_tape[1]) > 0:
                    best_epoch = np.argmax(valid_scores)
                    
                    best_train = train_tape[1][best_epoch]
                    best_valid = valid_tape[1][best_epoch]
                    best_test  = test_tape[1][best_epoch]

                    all_configs_results.append((config_number, (best_train, best_valid, best_test)))

    # Sorts the results by config number
    all_configs_results = sorted(all_configs_results, key=operator.itemgetter(0))

    config_numbers = []
    train_scores = []
    valid_scores = []
    test_scores = []

    for results in all_configs_results:
        config_numbers.append(results[0])

        train_scores.append(results[1][0])
        valid_scores.append(results[1][1])
        test_scores.append(results[1][2])

    # Plot
    locations = np.arange(len(all_configs_results))  # the x locations for the groups
    width = 0.35  # the width of the bars

    plt.figure(figsize=(len(config_numbers), 10))
    ax = plt.subplot(1,1,1)

    bars1 = ax.bar(locations, train_scores, width, color='blue', label='Train')
    bars2 = ax.bar(locations + width, valid_scores, width, color='orange', label='Valid')
    bars3 = ax.bar(locations + 2*width, test_scores, width, color='purple', label='Test')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Comparative score chart for configs')
    ax.set_xticks(locations + width/3)
    ax.set_xticklabels(config_numbers)
    #ax.legend((rects1[0], rects2[0]), ('Valid', 'Test'))
    ax.legend(loc='best')


def saveComparativeBarChart(data_file, config_number, best_valid_acc, train_acc, test_acc):
    

    def autolabel(rects):
        # Attach a text label above each bar displaying its height
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 0.85 * height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.savefig(''.join(data_path.split('.')[:-1]) + '.png')
    plt.close()