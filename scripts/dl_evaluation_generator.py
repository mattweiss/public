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

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/dl_model.py'
#****************************************************************************************************************************
experiments = [
    ('lstm_cauchy_100k_sine',list(range(1,101))),
    #('aekf_allnoise_ncv_100k_taylor',[5]),
    #('aekf_allnoise_nca_100k_taylor',[25]),
    #('aekf_allnoise_jerk_100k_taylor',[16]),
]                  

test_dataset_files = [
    #'FUNC_taylor_poly_NOISE_gaussian_LOC_0_SCALE_1_TRIALS_100_SAMPLES_100_DOMAIN_minus1_1_FEATURES_1_N_10.pkl',
    #'FUNC_taylor_poly_NOISE_bimodal_LOC_3_SCALE_1_TRIALS_100_SAMPLES_100_DOMAIN_minus1_1_FEATURES_1_N_10.pkl',
    #'FUNC_taylor_poly_NOISE_cauchy_LOC_na_SCALE_na_TRIALS_100_SAMPLES_100_DOMAIN_minus1_1_FEATURES_1_N_10.pkl',
    'FUNC_sine_NOISE_cauchy_LOC_na_SCALE_na_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
]

machine = socket.gethostname()

#****************************************************************************************************************************

for experiment in experiments:

    experiment_name = experiment[0]
    model_ids = experiment[1]
     
    experiment_dir = '/Documents/wpi/research/code/dovebirdia/experiments/' + experiment_name + '/'
    test_dataset_dir = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/test_datasets/'

    for test_dataset_file in test_dataset_files:

        test_dataset_path = test_dataset_dir + test_dataset_file

        #######################
        # Write Config Files
        #######################

        for model_id in model_ids:

            # Create Directories
            model_dir_name = experiment_name + '_model_' + str(model_id) + '/'
            model_dir = os.environ['HOME'] + experiment_dir + model_dir_name

            # bash-batch script
            if machine == 'pengy':

                batch_string_prefix = 'python3 '

            else:

                batch_string_prefix = 'sbatch -o ./testing_results.out '

            batch_str = batch_string_prefix + script + ' -d ' + test_dataset_path + '\n'
            batch_file_name = model_dir + 'test_model.sh'
            batch_file = open(batch_file_name, 'a')
            batch_file.write(batch_str)
            batch_file.close()
