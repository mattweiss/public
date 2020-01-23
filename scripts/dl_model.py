#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00

################################################################################
# DESCRIPTION
################################################################################

################################################################################
# MODULES
################################################################################

# Python
import numpy as np
import os, sys
import dill
from datetime import datetime
import argparse
from pdb import set_trace as st
import csv
import pandas as pd

from dovebirdia.deeplearning.networks.autoencoder import AutoencoderKalmanFilter
from dovebirdia.deeplearning.networks.lstm import LSTM
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.datasets.nyse_dataset import nyseDataset

################################################################################
# PROCESS COMMAND LINE FLAGS
################################################################################

# define command line parser
parser = argparse.ArgumentParser()

# if parser.parse_args().__dict__['training'] == 'True':

#     training_flag = True

# else:

#     training_flag = False

# get train/test flag
parser.add_argument("-c", "--config", dest = "config", help="Configuration File")
parser.add_argument("-r", "--results", dest = "results", help="Results q File Description")
parser.add_argument("-d", "--dataset", dest = "dataset", help="Test Dataset Path")

# parse options from commandline and copy to dictionary
flag_dict = parser.parse_args().__dict__

if flag_dict['dataset'] == None:

    TRAINING = True

else:

    TRAINING = False

if TRAINING:

    # read flag_dict values
    config_dir = flag_dict['config']

    # read all config files
    config_files = os.listdir(config_dir)

    config_dicts = dict()
    for config_file in config_files:

        config_name = os.path.splitext(config_file)[0].split('_')[-1]

        with open(config_dir + config_file, 'rb') as handle:

            config_dicts[config_name] = dill.load(handle)

else:

    test_dataset_path = flag_dict['dataset']

if TRAINING:

    ################################################################################
    # Create Directories
    ################################################################################

    # create results directory
    current_time = datetime.now().strftime("%m%d-%H-%M-%S")
    current_day = datetime.now().strftime("%m%d") + '/'
    res_dir = flag_dict['results']

    res_file = res_dir.split('/')[-2]

    if not os.path.exists(res_dir):

        os.makedirs( res_dir )

else:

    ################################################################################
    # Load Model Data From config Directory
    ################################################################################

    # read all config files
    config_dir = './config/'
    config_files = os.listdir(config_dir)

    config_dicts = dict()
    for config_file in config_files:

        config_name = os.path.splitext(config_file)[0].split('_')[-1]

        with open(config_dir + config_file, 'rb') as handle:

            print(config_dir + config_file)

            config_dicts[config_name] = dill.load(handle)

    ################################################################################
    # Load Test Dataset
    ################################################################################

    if config_dicts['test']['dataset'] == 'DomainRandomizationDataset':

        dataset = DomainRandomizationDataset(config_dicts['ds']).getDataset(test_dataset_path)

    elif config_dicts['test']['dataset'] == 'nyseDataset':

        dataset = nyseDataset(config_dicts['test']).getDataset()

    # data from domain randomization tests
    try:

        #x_test, y_test, t = dataset['data']['x_test'], dataset['data']['y_test'], dataset['data']['t']
        x_test, y_test = dataset['data']['x_test'], dataset['data']['y_test']

    # existing dataset (i.e. weather, stock market)
    except:

        # x_test, y_test, t = np.expand_dims(dataset['x_test'],axis=-1), np.expand_dims(dataset['x_test'],axis=-1), dataset['t']
        x_test, y_test = np.expand_dims(dataset['x_test'],axis=-1), np.expand_dims(dataset['x_test'],axis=-1)

################################################################################
# Model
################################################################################

# Hidden dimensions
config_dicts['model']['hidden_dims'] = list(config_dicts['model']['hidden_dims'])

# if using AEKF
if config_dicts['meta']['network'].__name__ == 'AutoencoderKalmanFilter':

    # append n_signals
    config_dicts['model']['hidden_dims'].append(config_dicts['kf']['n_signals'])

    nn = config_dicts['meta']['network'](config_dicts['model'], config_dicts['kf'])

elif config_dicts['meta']['network'].__name__ == 'HilbertAutoencoderKalmanFilter':

    nn = config_dicts['meta']['network'](config_dicts['model'], config_dicts['kf'])

else:

    nn = config_dicts['meta']['network'](config_dicts['model'])

print(nn.__class__)
nn.compile()

if TRAINING:

    # aekf
    try:

        # append either F and H to config_dicts to be saved to csv if either are random
        if config_dicts['kf']['f_model'] == 'random':

            config_dicts['kf']['F'] = nn.__dict__['_kalman_filter']._F

        if config_dicts['kf']['h_model'] == 'random':

            config_dicts['kf']['H'] = nn.__dict__['_kalman_filter']._H

    except:

        pass

    history = nn.fit(dr_params=config_dicts['ds'], save_model=True)
    history_size = config_dicts['model']['history_size']
    train_mse = np.asarray(history['train_loss'][-history_size:]).mean()
    train_std = np.asarray(history['train_loss'][-history_size:]).std()
    val_mse = np.asarray(history['val_loss'][-history_size:]).mean()
    val_std = np.asarray(history['val_loss'][-history_size:]).std()

    try:

        test_mse = np.asarray(history['test_loss']).mean()
        test_std = np.asarray(history['test_loss']).std()

    except:

        pass

    results_dict = {
        'train_mse':train_mse,
        'val_mse':val_mse,
        'test_mse':test_mse,
        'runtime':history['runtime'],
    }

    print('Training MSE, STD: {train_mse}, {train_std}\nValidation MSE, STD: {val_mse}, {val_std}\nTesting MSE, STD: {test_mse}, {test_std}'.format(train_mse=train_mse,
                                                                                                                                                    train_std=train_std,
                                                                                                                                                    val_mse=val_mse,
                                                                                                                                                    val_std=val_std,
                                                                                                                                                    test_mse=test_mse,
                                                                                                                                                    test_std=test_std))
else:

    evaluate_save_path = test_dataset_path.split('/')[-1].split('.')[0]
    history = nn.evaluate(x=x_test, y=y_test,
                          #t=t,
                          save_results=evaluate_save_path)

    results_dict = {
        'test_mse':np.asarray(history['test_loss']).mean(),
        'test_std':np.asarray(history['test_loss']).std(),
    }

################################################################################
# CSV
################################################################################

# Remove F and H from config_dicts['kf']
try:

    del config_dicts['kf']['F']
    del config_dicts['kf']['H']

except:

    pass

# merge dictionaries in config_dicts and training_results_dict
merged_config_dicts = dict()

if TRAINING:

    # config dictionaries
    for config_dict in config_dicts.values():

        merged_config_dicts.update(config_dict)

else:

    # dataset
    merged_config_dicts.update({'test_dataset':test_dataset_path})

# model id
merged_config_dicts.update({'model_id':os.getcwd().split('/')[-1].split('_')[-1]})

# training results

merged_config_dicts.update(results_dict)

# change dictionary value to name if exists
for k,v in merged_config_dicts.items():

    try:

        merged_config_dicts[k] = v.__name__

    except:

        pass

if TRAINING:

    results_file = 'training_results.csv'

else:

    results_file = 'testing_results.csv'

results_file_path = os.getcwd() + config_dicts['model']['results_dir'] + results_file

try:

    with open(results_file_path, 'a') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=sorted(merged_config_dicts.keys()))

        if os.stat(results_file_path).st_size == 0:

            writer.writeheader()

        writer.writerow(merged_config_dicts)

except IOError:

    print("I/O error")
