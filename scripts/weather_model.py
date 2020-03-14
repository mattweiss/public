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
from dovebirdia.datasets.weather_dataset import weatherDataset

################################################################################
# PROCESS COMMAND LINE FLAGS
################################################################################

# define command line parser
parser = argparse.ArgumentParser()

# get train/test flag
parser.add_argument("-c", "--config", dest = "config", help="Configuration File")
parser.add_argument("-r", "--results", dest = "results", help="Results q File Description")
parser.add_argument("-d", "--dataset", dest = "dataset", help="Test Dataset Path")

# parse options from commandline and copy to dictionary
flag_dict = parser.parse_args().__dict__

# read flag_dict values
config_dir = flag_dict['config']

# read all config files
config_files = os.listdir(config_dir)

config_dicts = dict()
for config_file in config_files:

    config_name = os.path.splitext(config_file)[0].split('_')[-1]

    with open(config_dir + config_file, 'rb') as handle:

        config_dicts[config_name] = dill.load(handle)

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

################################################################################
# Load Test Dataset
################################################################################

dataset = weatherDataset(config_dicts['ds']).getDataset()

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

history = nn.fit(dataset=dataset, save_model=True)

train_mse = np.asarray(history['train_loss'][-1])
val_mse = np.asarray(history['val_loss'][-1])
test_mse = np.asarray(history['test_loss'][-1])

print('Training MSE: {train_mse:0.4}\nValidation MSE: {val_mse:0.4}\nTesting MSE: {test_mse:0.4}'.format(train_mse=train_mse,
                                                                                                        val_mse=val_mse,
                                                                                                        test_mse=test_mse))

results_dict = {
    'train_mse':train_mse,
    'val_mse':val_mse,
    'test_mse':test_mse,
    'runtime':history['runtime'],
}

# save test truth and predictions to disk
#test_pred_file_path = os.getcwd() + config_dicts['model']['results_dir'] + 'test_pred'
#np.save(test_pred_file_path,history['test_pred'])

#test_true_file_path = os.getcwd() + config_dicts['model']['results_dir'] + 'test_true'
#np.save(test_true_file_path,history['test_true'])

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

# config dictionaries
for config_dict in config_dicts.values():

    merged_config_dicts.update(config_dict)

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

results_file = 'training_results.csv'

results_file_path = os.getcwd() + config_dicts['model']['results_dir'] + results_file

try:

    with open(results_file_path, 'a') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=sorted(merged_config_dicts.keys()))

        if os.stat(results_file_path).st_size == 0:

            writer.writeheader()

        writer.writerow(merged_config_dicts)

except IOError:

    print("I/O error")
