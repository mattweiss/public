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
import os
import dill
from datetime import datetime
import argparse
from pdb import set_trace as st
import csv
import pandas as pd

from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.filtering.kalman_filter import KalmanFilter

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

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

################################################################################
# Dataset
################################################################################

dataset = DomainRandomizationDataset(config_dicts['dr']).getDataset(config_dicts['dr']['load_path'])
x_test, y_test, t = dataset['data']['x_test'], dataset['data']['y_test'], dataset['data']['t']

################################################################################
# Model
################################################################################

filter = config_dicts['meta']['filter'](config_dicts['kf'])
print(filter.__class__)

history = filter.evaluate(x_test,y_test,t,with_np=True)

sw = np.asarray(history['sw'])

# plot of Shapiro-Wilk results
plt.figure(figsize=(6,6))
plt.scatter(range(sw.shape[0]),sw)
plt.title('{info}\n{mean_mse}'.format(info=config_dicts['dr']['load_path'].split('/')[-1].split('_')[:4],mean_mse=sw.mean()))
plt.xlabel('Sample')
plt.ylabel('SW p-value')
plot_file_path = os.getcwd() + config_dicts['model']['results_dir'] + config_dicts['dr']['load_path'].split('/')[-1].split('.')[0] + '_shapiro-wilk'
plt.savefig(plot_file_path)
#plt.show()
plt.close()

sw_min = sw.min()
sw_max = sw.max()

n_sw_lt_min = sw[sw < (0.05/sw.shape[0])].shape[0]
per_sw_lt_min = float(n_sw_lt_min / sw.shape[0]) * 100

results_dict = {
    'test_mse':np.asarray(history['test_loss']).mean(),
    'sw_min':sw_min,
    'sw_max':sw_max,
    'n_sw_lt_min':n_sw_lt_min,
    'per_sw_lt_min':per_sw_lt_min,
}

################################################################################
# CSV
################################################################################

# merge dictionaries in config_dicts and training_results_dict
merged_config_dicts = dict()

for config_dict in config_dicts.values():

    merged_config_dicts.update(config_dict)

# training results
merged_config_dicts.update(results_dict)

# model id
merged_config_dicts.update({'model_id':os.getcwd().split('/')[-1].split('_')[-1]})

# change dictionary value to name if exists
for k,v in merged_config_dicts.items():

    try:

        merged_config_dicts[k] = v.__name__

    except:

        pass
        
results_file = os.getcwd() + config_dicts['model']['results_dir'] + 'testing_results.csv'

try:
    
    with open(results_file, 'a') as csvfile:
        
        writer = csv.DictWriter(csvfile, fieldnames=sorted(merged_config_dicts.keys()))

        if os.stat(results_file).st_size == 0:

            writer.writeheader()

        writer.writerow(merged_config_dicts)

except IOError:

    print("I/O error") 

