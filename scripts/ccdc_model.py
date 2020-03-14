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
config_dicts['model']['input_dim'] = (config_dicts['dataset']['samples'][1]-config_dicts['dataset']['samples'][0])*n_sensors

nn = config_dicts['meta']['network'](config_dicts['model'])

history = nn.fit(dataset)

train_loss = np.asarray(history['loss'][-1])
val_loss = np.asarray(history['val_loss'][-1])
test_loss = np.asarray(history['test_loss'])
test_subset_accuracy = history['test_subset_accuracy']
val_subset_accuracy = history['val_subset_accuracy']


print('Training Loss: {train_loss:0.4}\nValidation Loss: {val_loss:0.4}\nTesting Loss: {test_loss:0.4}'.format(train_loss=train_loss,
                                                                                                               val_loss=val_loss,
                                                                                                               test_loss=test_loss))

print('Validation Accuracy: {val_subset_accuracy:0.4}\nTesting Accuracy: {test_subset_accuracy:0.4}'.format(val_subset_accuracy=val_subset_accuracy,
                                                                                                            test_subset_accuracy=test_subset_accuracy))


n_labels = 5
print('First {n_labels} Test Truth and Predictions:\n{test_true}\n\n{test_pred}'.format(n_labels=n_labels,
                                                                                       test_true=history['test_true'][:n_labels],
                                                                                       test_pred=history['test_pred'][:n_labels]))
        

results_dict = {
    'train_loss':train_loss,
    'val_loss':val_loss,
    'test_loss':test_loss,
    'test_subset_accuracy':test_subset_accuracy,
    'val_subset_accuracy':val_subset_accuracy,
    'runtime':history['runtime'],
}

# save test predictions to disk
test_pred_file_path = os.getcwd() + config_dicts['model']['results_dir'] + 'test_pred'
test_true_file_path = os.getcwd() + config_dicts['model']['results_dir'] + 'test_true'
np.save(test_pred_file_path,history['test_pred'])
np.save(test_true_file_path,history['test_true'])

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
