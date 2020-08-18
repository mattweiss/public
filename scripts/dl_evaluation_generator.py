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
project = 'dissertation/ccdc'

config_dict = dict()
#config_dict['dataset'] = 'DomainRandomizationDataset'
#config_dict['dataset'] = 'FlightKinematicsDataset'
config_dict['dataset'] = 'CCDCDataset'

config_dict['with_val'] = False

experiments = [
    # ('aekf_turns_1_cauchy_0_5_F_NCA_Q_0-5',[25]),
    # ('aeimm_turns_1_cauchy_0_5_F_NCA1_NCA2_Q_0-5',[4]),
    # ('lstm_turns_1_cauchy_0_5',[37])
    #('aeimm_turns_1_gaussian_0_20_F_NCA_Q_0-5',[2]),
    #('aekf_turns_1_gaussian_0_20_F_NCA_Q_0-5',[3]),
    #('lstm_turns_1_gaussian_0_20',[18])

    ('aekf_dim_8_curve_taylor_Noise_gaussian_F_NCV_N_3_7_R_learned_epoch_10k_features_1_train_ground_samples_100_act_leaky',[13])
    ]

test_dataset_files = [
    #'/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/dissertation/imm/eval/benchmark_gaussian_20_turn.pkl',
    #'/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark/split/01_23_19/testing/01232019_vaportest_trial_107_label_12.pkl'
    '/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark/split/01_23_19/validation/01232019_vaportest_trial_247_label_21.pkl'
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
