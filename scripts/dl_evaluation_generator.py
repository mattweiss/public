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
project = 'asilomar'

experiments = [

    #('aekf_gaussian_100k_ncv_taylor',[14]),
    #('aekf_bimodal_100k_ncv_taylor',[15]),
    #('aekf_cauchy_100k_ncv_taylor',[20]),
    #('lstm_gaussian_100k_taylor',[98]),
    #('lstm_bimodal_100k_taylor',[41]),
    ('lstm_cauchy_100k_taylor',[42]),

]

test_dataset_files = [

    # asilomar
    #'FUNC_taylor_NOISE_gaussian_LOC_0_SCALE_0-2_TRIALS_1000_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_3.pkl',
    #'FUNC_taylor_NOISE_bimodal_LOC_0-25_SCALE_0-2_TRIALS_1000_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_3.pkl',
    'FUNC_taylor_NOISE_cauchy_LOC_na_SCALE_na_TRIALS_1000_SAMPLES_100_DOMAIN_0_100_FEATURES_1_N_3.pkl',
    
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
