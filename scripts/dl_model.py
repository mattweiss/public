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

            config_dicts[config_name] = dill.load(handle)

    ################################################################################
    # Load Test Dataset
    ################################################################################

    config_dicts['dr']['load_path'] = test_dataset_path
    dataset = DomainRandomizationDataset(config_dicts['dr']).getDataset()

    x_test, y_test, t = dataset['data']['x_test'], dataset['data']['y_test'], dataset['data']['t']

################################################################################
# Model
################################################################################

# Network
config_dicts['model']['hidden_dims'] = list(config_dicts['model']['hidden_dims'])


# if using AEKF
if isinstance(config_dicts['meta']['network'], AutoencoderKalmanFilter):
    
    config_dicts['model']['hidden_dims'].append(config_dicts['kf']['n_signals'])
    nn = config_dicts['meta']['network'](config_dicts['model'], config_dicts['kf'])

else:

    nn = config_dicts['meta']['network'](config_dicts['model'])
    
print(nn.__class__)

if TRAINING:

    history = nn.fitDomainRandomization(config_dicts['dr'], save_model=True)

    results_dict = {
        'train_mse':np.asarray(history['train_loss']).mean(),
        'val_mse':np.asarray(history['val_loss']).mean()
        }

else:

    history = nn.evaluate(x=x_test, y=y_test, t=t)
    
    results_dict = {
    'test_mse':np.asarray(history['test_loss']).mean(),
    }
    
################################################################################
# CSV
################################################################################

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

