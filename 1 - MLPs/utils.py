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

def save_results(train_tape, valid_tape, test_tape, save_dir, exp_name, data_file, show_test):

    # Creates the folder if necessary
    saving_dir = os.path.join(save_dir, exp_name)
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    best_epoch = np.argmax(valid_tape[1])

    best_train_acc = train_tape[1][best_epoch]
    best_valid_acc = valid_tape[1][best_epoch]
    best_test_acc = test_tape[1][best_epoch]

    last_train_acc = train_tape[1][-1]
    last_valid_acc = valid_tape[1][-1]
    last_test_acc = test_tape[1][-1]

    # Creates and save the plots
    plt.figure(figsize=(20, 6))

    n_epochs = len(valid_tape[0])
    epochs = np.arange(n_epochs)

    plt.subplot(1,2,1)
    plt.title("Loss", fontweight='bold')
    plt.plot(epochs, train_tape[0], color="blue", label="Training set")
    plt.plot(epochs, valid_tape[0], color="orange", label="Validation set")
    if show_test:
        plt.plot(epochs, test_tape[0], color="purple", label="Test set")
    plt.xlabel("Epochs")
    plt.legend(loc='best')

    plt.subplot(1,2,2)
    plt.title("Accuracy", fontweight='bold')
    plt.plot(epochs, train_tape[1], color="blue", label="Training set, best={0:.2f}, last={1:.2f}".format(best_train_acc, last_train_acc))
    plt.plot(epochs, valid_tape[1], color="orange", label="Validation set, best={0:.2f}, last={1:.2f}".format(best_valid_acc, last_valid_acc))
    if show_test:
        plt.plot(epochs, test_tape[1], color="purple", label="Test set, best={0:.2f}, last={1:.2f}".format(best_test_acc, last_test_acc))
    plt.ylim(0, 100)
    plt.xlabel("Epochs")
    plt.legend(loc='best')

    plt.axvline(x=best_epoch)

    plt.savefig(os.path.join(saving_dir, exp_name + '.png'), bbox_inches='tight')
    plt.close()

    # Save the recording tapes (learning curves) in a file
    log_file = os.path.join(saving_dir, 'log_' + exp_name + '.pkl')
    with open(log_file, 'wb') as f:
        pickle.dump({
            'data_file': data_file,
            'train_tape': train_tape,
            'valid_tape': valid_tape,
            'test_tape': test_tape
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Results saved')

    return


def update_comparative_chart(result_dir, show_test):
    
    all_configs_results = []

    # Finds the right folders
    result_folders = os.listdir(result_dir)

    for thing in result_folders:
        
        if len(thing.split('.')) == 1 and (thing.startswith('config') or thing.startswith('sample')): # if thing is a folder

            folder = thing
            if folder.startswith('config'):
                things_name = 'configs'
                config_number = int(folder.lstrip('config'))
            elif folder.startswith('sample'):
                things_name = 'samples'
                config_number = int(folder.lstrip('sample'))
            files = os.listdir(os.path.join(result_dir, folder))
            for f in files:
                if f.startswith('log_'):
                    log_file = os.path.join(result_dir, folder, f)

            # If the log_file exists, extracts the recording tapes it contains
            if os.path.exists(log_file):
                
                with open(log_file, 'rb') as f:
                    log_data = pickle.load(f)
                
                data_file = log_data['data_file']
                
                train_tape = log_data['train_tape']
                valid_tape = log_data['valid_tape']
                test_tape  = log_data['test_tape']
                
                # If tape is not empty, collects the results
                if len(valid_tape[1]) > 0:
                    best_epoch = np.argmax(valid_tape[1])
                    
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
    if show_test:
        n_bars = 3
        bar_width = .75
    else:
        n_bars = 2
        bar_width = 1.15

    locations = 3 * np.arange(len(all_configs_results))  # the x locations for the groups

    plt.figure(figsize=(np.max(locations), 10))
    ax = plt.subplot(1,1,1)

    bars1 = ax.bar(locations, train_scores, bar_width, color='blue', label='Train')
    bars2 = ax.bar(locations + bar_width, valid_scores, bar_width, color='orange', label='Valid')
    if show_test:
        bars3 = ax.bar(locations + 2*bar_width, test_scores, bar_width, color='purple', label='Test')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparative score chart for {}'.format(things_name), fontweight='bold')
    ax.set_xticks(locations + ((n_bars-1)*bar_width/2.)) # (n_bars-1)*
    ax.set_xticklabels(config_numbers)
    ax.legend(loc='best')


    def autolabel(bars):
        # Attach a text label above each bar displaying its height
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 0.85 * height, '%.2f' % height, ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    if show_test:
        autolabel(bars3)

    plt.savefig(os.path.join(result_dir, 'results.png'))
    plt.close()

    print("Comparative chart has been updated")

    return