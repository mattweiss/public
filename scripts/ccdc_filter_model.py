#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32G
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
import tensorflow as tf
import os
import dill
from datetime import datetime
import argparse
from pdb import set_trace as st
import csv
import pandas as pd

from dovebirdia.datasets.ccdc_mixtures import ccdcMixturesDataset
from dovebirdia.filtering.kalman_filter import KalmanFilter
from dovebirdia.utilities.base import saveDict

################################################################################
# PROCESS COMMAND LINE FLAGS
################################################################################

# define command line parser
parser = argparse.ArgumentParser()

# -cfg CONFIG_FILE
parser.add_argument("-c", "--config", dest = "config", help="Configuration File")
parser.add_argument("-r", "--results", dest = "results", help="Results q File Description")

# parse options from commandline and copy to dictionary
flag_dict = parser.parse_args().__dict__

# display config_diction
for k,v in flag_dict.items():

    print('{}: {}'.format(k,v))

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
# SET DIRECTORIES
################################################################################

# create results directory
current_time = datetime.now().strftime("%m%d-%H-%M-%S")
current_day = datetime.now().strftime("%m%d") + '/'
res_dir = flag_dict['results']

res_file = res_dir.split('/')[-2]
# print('Results file: %s' % (res_file))

if not os.path.exists(res_dir):

    os.makedirs(res_dir)

# create new split directory for Kalman Filtered data

kf_split_base_dir = config_dicts['dataset']['dataset_dir'][:-1] + \
                    '_kf_q_' + str(config_dicts['kf']['q']).replace('.','-') + \
                    '_r_' + str(config_dicts['kf']['r']).replace('.','-') + \
                    config_dicts['dataset']['dataset_dir'][-1]

try:
    
    os.makedirs(kf_split_base_dir + 'training')
    os.makedirs(kf_split_base_dir + 'validation')
    os.makedirs(kf_split_base_dir + 'testing')

except:
    
    pass

################################################################################
# Filter and save to directory
################################################################################

trials_to_filter = None

split_list = [
    'training',
    'validation',
    'testing'
    ]

for split_name in split_list:

    # load pickle files into dataframe
    split_dir = config_dicts['dataset']['dataset_dir'] + split_name + '/'
    pickle_files = os.listdir(split_dir)

    # loop over each pickle file
    for pickle_file in pickle_files[:trials_to_filter]:

        # load df
        df = np.load(split_dir+pickle_file, allow_pickle=True)

        z = df['resistance'][config_dicts['dataset']['samples'][0]:config_dicts['dataset']['samples'][1],:]

        filter = config_dicts['meta']['filter'](config_dicts['kf'])
        filter_results = filter.fit(z)
        
        x_hat_post = np.squeeze(filter_results['x_hat_post'],axis=-1)

        tf.reset_default_graph()

        # add new key to dictionary
        df['resistance_kf0'] = x_hat_post[:,::2]
        df['resistance_kf1'] = x_hat_post[:,1::2]
        
        # write dictionary to pickle file
        dill_output_file = kf_split_base_dir + split_name + '/' + pickle_file
        with open(dill_output_file, 'wb') as handle:

            dill.dump(df, handle, protocol=dill.HIGHEST_PROTOCOL)

################################################################################
# CSV
################################################################################

# merge dictionaries in config_dicts and training_results_dict
merged_config_dicts = dict()

# remove F and H from dictionary before writing to csv
config_dicts['kf'].pop('F')
config_dicts['kf'].pop('H')

for config_dict in config_dicts.values():

    merged_config_dicts.update(config_dict)

# change dictionary value to name if exists
for k,v in merged_config_dicts.items():

    try:

        merged_config_dicts[k] = v.__name__

    except:

        pass
        
results_file = kf_split_base_dir + 'kf_params.csv'

try:
    
    with open(results_file, 'a') as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=sorted(merged_config_dicts.keys()))

        if os.stat(results_file).st_size == 0:

            writer.writeheader()

        writer.writerow(merged_config_dicts)

except IOError:

    print("I/O error") 

