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

from dovebirdia.utilities.base import loadDict
from dovebirdia.deeplearning.networks.autoencoder import AutoencoderKalmanFilter, AutoencoderInteractingMultipleModel
from dovebirdia.deeplearning.networks.lstm_tf import LSTM
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.datasets.nyse_dataset import nyseDataset

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
    
    dataset = loadDict(test_dataset_path)
    
    # data from domain randomization tests
    if config_dicts['test']['dataset'] == 'DomainRandomizationDataset' or config_dicts['test']['dataset'] == 'FlightKinematicsDataset':

        x_test, y_test = dataset['data']['x_test'], dataset['data']['y_test']
        
    # existing dataset (i.e. weather, stock market, s5)
    elif config_dicts['test']['dataset'] == 'nyseDataset' or \
         config_dicts['test']['dataset'] == 's5Dataset' or \
         config_dicts['test']['dataset'] == 'weatherDataset':
    
        x_test, y_test = dataset['data']['x_test'], dataset['data']['x_test']

    # pets dataset
    elif config_dicts['test']['dataset'] == 'petsDataset' or \
         config_dicts['test']['dataset'] == 'mtrDataset':

        x_test, y_test = dataset['data']['x_test'], dataset['data']['x_true']

    # if labels exist
    try:

        labels = dataset['data']['y_test']

    except:

        labels = None
        
################################################################################
# Model
################################################################################

# Hidden dimensions
config_dicts['model']['hidden_dims'] = list(config_dicts['model']['hidden_dims'])

# if using AEKF
#if config_dicts['meta']['network'].__name__ == 'AutoencoderKalmanFilter' or 'AutoencoderInteractingMultipleModel':
try:
    
    # append n_signals
    config_dicts['model']['hidden_dims'].append(config_dicts['kf']['meas_dims'])
    nn = config_dicts['meta']['network'](config_dicts['model'], config_dicts['kf'])

#else:
except:
    
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

    train_loss = np.asarray(history['train_loss']).mean()
    train_loss_std = np.asarray(history['train_loss']).std()
    train_mse = np.asarray(history['train_mse']).mean()
    train_mse_std = np.asarray(history['train_mse']).std()

    val_loss = np.asarray(history['val_loss']).mean()
    val_std = np.asarray(history['val_loss']).std()
    val_mse = np.asarray(history['val_mse']).mean()
    val_mse_std = np.asarray(history['val_mse']).std()

    metric_sublen = dr_params=config_dicts['ds']['metric_sublen']

    train_loss_sub = np.asarray(history['train_loss'][-metric_sublen:]).mean()
    train_loss_std_sub = np.asarray(history['train_loss'][-metric_sublen:]).std()
    train_mse_sub = np.asarray(history['train_mse'][-metric_sublen:]).mean()
    train_mse_std_sub = np.asarray(history['train_mse'][-metric_sublen:]).std()

    val_loss_sub = np.asarray(history['val_loss'][-metric_sublen:]).mean()
    val_std_sub = np.asarray(history['val_loss'][-metric_sublen:]).std()
    val_mse_sub = np.asarray(history['val_mse'][-metric_sublen:]).mean()
    val_mse_std_sub = np.asarray(history['val_mse'][-metric_sublen:]).std()

    results_dict = {
        'train_loss':train_loss,
        'val_loss':val_loss,
        'train_mse':train_mse,
        'val_mse':val_mse,
        'train_loss_sub':train_loss_sub,
        'val_loss_sub':val_loss_sub,
        'train_mse_sub':train_mse_sub,
        'val_mse_sub':val_mse_sub,
        'runtime':history['runtime'],
    }
   
else:

    evaluate_save_path = test_dataset_path.split('/')[-1].split('.')[0]

    class_name = type(nn).__name__

    # default eval ops and attributes lists
    eval_ops_list =  None
    attributes_list = None
        
    if class_name == 'AutoencoderKalmanFilter' or class_name ==  'AutoencoderInteractingMultipleModel':

        eval_ops_list =  ['kf_results']
        
    elif class_name == 'LSTM':

        attributes_list = ['seq_len']

    history = nn.evaluate(x=x_test, y=y_test,labels=labels,
                          eval_ops=eval_ops_list,
                          attributes=attributes_list,
                          save_results=evaluate_save_path)

    results_dict = {
        'test_loss':np.asarray(history['loss_op']).mean(),
        'test_loss_std':np.asarray(history['loss_op']).std(),
        'test_mse':np.asarray(history['mse_op']).mean(),
        'test_mse_std':np.asarray(history['mse_op']).std(),
    }

################################################################################
# CSV
################################################################################

# Write Kalman Filter Parameters to Separate File
try:

    fo = open(os.getcwd() + config_dicts['model']['results_dir'] + 'kf_params.txt', 'w')

    for k, v in config_dicts['kf'].items():

        fo.write('# '+ str(k) + '\n\n' + str(v) + '\n\n')

    fo.close()

    del config_dicts['kf']
    # del config_dicts['kf']['F']
    # del config_dicts['kf']['H']
    # config_dicts['kf'].update({'Q':np.unique(config_dicts['kf']['Q'])})
    
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

    results_file = os.getcwd().split('/')[-1] + '_training_results.csv'

else:

    results_file = os.getcwd().split('/')[-1] + '_testing_results.csv'

results_file_path = os.getcwd() + config_dicts['model']['results_dir'] + results_file

try:

    with open(results_file_path, 'a') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=sorted(merged_config_dicts.keys()))

        if os.stat(results_file_path).st_size == 0:

            writer.writeheader()

        writer.writerow(merged_config_dicts)

except IOError:

    print("I/O error")
