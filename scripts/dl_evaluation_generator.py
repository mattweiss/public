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

from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from dovebirdia.datasets.nyse_dataset import nyseDataset

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/dl_model.py'
#****************************************************************************************************************************
project = 'nyse'

config_dict = dict()
config_dict['dataset'] = 'DomainRandomizationDataset'
#config_dict['dataset'] = 'FlightKinematicsDataset'
config_dict['with_val'] = False

experiments = [
    ('aekf_dim_8_curve_legendre_Noise_gaussian_F_NCA_N_20_30_R_learned_epoch_100k_features_1_train_ground_samples_350_act_leaky',[6]),
]

test_dataset_files = [
    #'/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/imm/eval/benchmark.pkl',
    '/home/mlweiss/Documents/wpi/research/data/nyse/split/nyse_all_train_test_split_n_securities_None_n_samples_None_features_1.pkl',
    #'/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/nyse/eval/benchmark_legendre_cauchy_R1_1k.pkl'
]

machine = socket.gethostname()

#****************************************************************************************************************************

for experiment in experiments:

    experiment_name = experiment[0]
    model_ids = experiment[1]

    experiment_dir = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/' + project + '/' + experiment_name + '/'

    for test_dataset_file in test_dataset_files:

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
