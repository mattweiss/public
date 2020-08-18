#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=20G
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

from dovebirdia.deeplearning.networks.keras_classifiers import KerasMultiLabelClassifier
from dovebirdia.datasets.ccdc_mixtures import ccdcMixturesDataset

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
# Load Dataset
################################################################################

dataset = ccdcMixturesDataset(params=config_dicts['dataset']).getDataset()

################################################################################
# Model
################################################################################

# add output dims
n_sensors = 20 if config_dicts['dataset']['sensors'] is None else len(config_dicts['dataset']['sensors'])
config_dicts['model']['input_dim'] = (config_dicts['dataset']['samples'][1]-config_dicts['dataset']['samples'][0])*n_sensors if config_dicts['dataset']['pca_components']==0 else config_dicts['dataset']['pca_components']

nn = config_dicts['meta']['network'](config_dicts['model'])
#history = nn.fit(dataset)
syn_dataset_dir = '/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark/split/01_23_19/training/'
history = nn.fitDomainRandomization(data_dir=syn_dataset_dir,
                                    dataset=dataset)

for k,v in history.items():

    if 'error' in k:
        
        print(k,v)
        
results_dict = history

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
