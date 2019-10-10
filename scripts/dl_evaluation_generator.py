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
project = 'sdm'

experiments = [

    #('aekf_allnoise_100k_ncv_allcurves',[18]),
    ('aekf_allnoise_100k_nca_allcurves',[28]),
    ('aekf_allnoise_100k_jerk_allcurves',[43]),
    # ('lstm_allnoise_100k_allcurves',[19,39,59,80,95])

]                  

test_dataset_files = [

    # exponential
    'FUNC_exp_NOISE_bimodal_LOC_3_SCALE_1_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
    'FUNC_exp_NOISE_cauchy_LOC_na_SCALE_na_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
    'FUNC_exp_NOISE_gaussian_LOC_0_SCALE_5_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
    
    
    # sigmoid
    'FUNC_sig_NOISE_bimodal_LOC_3_SCALE_1_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
    'FUNC_sig_NOISE_cauchy_LOC_na_SCALE_na_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
    'FUNC_sig_NOISE_gaussian_LOC_0_SCALE_5_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',

    # sine
    'FUNC_sine_NOISE_bimodal_LOC_3_SCALE_1_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
    'FUNC_sine_NOISE_cauchy_LOC_na_SCALE_na_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
    'FUNC_sine_NOISE_gaussian_LOC_0_SCALE_5_TRIALS_100_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_na.pkl',
]

machine = socket.gethostname()

#****************************************************************************************************************************

for experiment in experiments:

    experiment_name = experiment[0]
    model_ids = experiment[1]
     
    experiment_dir = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/' + project + '/' + experiment_name + '/'
    test_dataset_dir = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/evaluation/' + project + '/'

    for test_dataset_file in test_dataset_files:

        test_dataset_path = test_dataset_dir + test_dataset_file

        #######################
        # Write Config Files
        #######################

        for model_id in model_ids:

            # Create Directories
            model_dir_name = experiment_name + '_model_' + str(model_id) + '/'
            model_dir = experiment_dir + model_dir_name

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
