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
project = 'nyse'

config_dict = dict()
config_dict['dataset'] = 'nyseDataset'
config_dict['saved_dataset'] = '/home/mlweiss/Documents/wpi/research/data/nyse/split/nyse_all_train_test_split.pkl'

experiments = [

    #('aekf_bimodal_100_ncv_taylor_fixed_F_fixed_H',[24]),
    #('aekf_bimodal_100_ncv_taylor_random_F_fixed_H',[32]),
    #('aekf_bimodal_100_ncv_taylor_learned_F_fixed_H',[28]),
    #('aekf_bimodal_100_ncv_taylor_fixed_F_random_H',[13]),
    #('aekf_bimodal_100_ncv_taylor_random_F_random_H',[69]),
    #('aekf_bimodal_100_ncv_taylor_learned_F_random_H',[23]),
    #('aekf_bimodal_100_ncv_taylor_fixed_F_learned_H',[30]),
    #('aekf_bimodal_100_ncv_taylor_random_F_learned_H',[68]),
    #('aekf_bimodal_100_ncv_taylor_learned_F_learned_H',[72]),
    ('aekf_nyse_dr_legendre_beta',[28,39]),

]

test_dataset_files = [

    # 'FUNC_taylor_NOISE_bimodal_LOC_0-05_SCALE_0-1_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_3_7.pkl'
    '/home/mlweiss/Documents/wpi/research/data/nyse/split/nyse_all_train_test_split.pkl'

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
