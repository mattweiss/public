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
import dovebirdia.utilities.distributions as distributions

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/dl_model.py'
experiment_name = 'aekf_gaussian_KILLME_taylor'
experiment_dir = '/Documents/wpi/research/code/dovebirdia/experiments/' + experiment_name + '/'
model_ids = [1,2]
test_dataset_path = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/datasets/test.pkl'
machine = socket.gethostname()
####################################

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

        batch_string_prefix = 'sbatch -o ./out/' + out_file_name + ' '
        
    batch_str = batch_string_prefix + script + ' -d ' + test_dataset_path + '\n'
    batch_file_name = model_dir + 'test_model.sh'
    batch_file = open(batch_file_name, 'w')
    batch_file.write(batch_str)
    batch_file.close()
