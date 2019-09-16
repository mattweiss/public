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
import mlflow

from dovebirdia.deeplearning.networks.autoencoder import AutoencoderKalmanFilter

# set tracking uri if it do not exist.
# mlflow occationally gives the following error: "File exists: '/home/mlweiss/Documents/wpi/research/code/dovebirdia/models/mlruns"
mlflow_dir = "/".join(os.getcwd().split('/')[:-2]) + '/mlruns'

if not os.path.isdir(mlflow_dir):

    mlflow_uri = 'file:' + mlflow_dir
    mlflow.set_tracking_uri(mlflow_uri)

mlflow.set_experiment(os.getcwd().split('/')[-2])
mlflow.start_run(run_name=os.getcwd().split('/')[-1])

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

    os.makedirs( res_dir )

################################################################################
# Model
################################################################################

# Network
config_dicts['model']['hidden_dims'] = list(config_dicts['model']['hidden_dims'])
config_dicts['model']['hidden_dims'].append(config_dicts['kf']['n_signals'])

#nn = config_dicts['meta']['model'](config_dicts['model'], config_dicts['kf'])
nn = AutoencoderKalmanFilter(config_dicts['model'], config_dicts['kf'])
print(nn.__class__)
nn.getModelSummary()
history = nn.fitDomainRandomization(config_dicts['dr'], save_model=True)

################################################################################
# mlflow
################################################################################

# params
for config_name, config_dict in config_dicts.items():

    for param_name, param in config_dict.items():

        try:

            mlflow.log_param(config_name + '_' + param_name, param.__name__)

        except:

            mlflow.log_param(config_name + '_' + param_name, param)

# metrics
for metric_name, metric in history.items():

    if len(metric) != 0:

        mlflow.log_metric(metric_name, np.asarray(metric).mean())
