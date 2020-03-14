#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:0

import os, sys, socket
import numpy as np
import itertools
import tensorflow as tf
import dill
import itertools
from collections import OrderedDict
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.autoencoder import AutoencoderKalmanFilter
import dovebirdia.utilities.dr_functions as drfns
import dovebirdia.stats.distributions as distributions

from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.datasets.nyse_dataset import nyseDataset

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/dl_model.py'
#****************************************************************************************************************************
project = 'pets'

config_dict = dict()
config_dict['dataset'] = 'petsDataset' #'DomainRandomizationDataset'
config_dict['with_val'] = False

experiments = [
    #('aekf_legendre_ncv_mask_percent_1_value_1000',[68]),
    #('aekf_legendre_real_ncv_mask_percent_1_value_1000',[70]),
    #('aekf_legendre_ncv_mask_percent_1_value_1000_KILLME',[1]),
    ('lstm_legendre_mask_percent_1_value_1000_KILLME',[1])
]

test_dataset_files = [
    '/home/mlweiss/Documents/wpi/research/data/pets/split/PETS2009_S2L1_dataset_SAMPLES_None_NOISE_none_MASK_percent_0-01_value_1000.pkl',
]

machine = socket.gethostname()

#****************************************************************************************************************************

for experiment in experiments:

    experiment_name = experiment[0]
    model_ids = experiment[1]

    experiment_dir = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/' + project + '/' + experiment_name + '/'
    # test_dataset_dir = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/evaluation/' + project + '/'

    for test_dataset_file in test_dataset_files:

        # test_dataset_path = test_dataset_dir + test_dataset_file
        test_dataset_path = test_dataset_file

        #######################
        # Write Config Files
        #######################

        for model_id in model_ids:

            # Create Directories
            model_dir_name = experiment_name + '_model_' + str(model_id) + '/'
            model_dir = experiment_dir + model_dir_name

            config_dict_path = model_dir + 'config/' + model_dir_name.split('/')[0] + '_test.cfg'

            with open(config_dict_path, 'wb') as handle:

                dill.dump(config_dict, handle)

            # bash-batch script
            if machine == 'pengy':

                batch_string_prefix = 'python3 '

            else:

                batch_string_prefix = 'sbatch -o ./testing_results.out '

            batch_str = batch_string_prefix + script + ' -c ./config' + ' -d ' + test_dataset_path + '\n'
            batch_file_name = model_dir + 'test_model.sh'
            batch_file = open(batch_file_name, 'a')
            batch_file.write(batch_str)
            batch_file.close()
