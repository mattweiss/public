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
from dovebirdia.filtering.kalman_filter import KalmanFilter
import dovebirdia.utilities.dr_functions as drfns 
import dovebirdia.stats.distributions as distributions

####################################
# Test Name and Description
####################################

script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/dtra_data_processing.py'

project = 'dtra_2019'

experiments = [
    ('kf_z_ncv',(1,2)),
]

machine = socket.gethostname()

########################
# Parameter dictionaries
########################

kf_params = dict()
gen_params = dict()

params_dicts = OrderedDict([
    ('gen',gen_params),
    ('kf',kf_params),
])
        
####################################
# General Parameters
####################################

gen_params['save_figures'] = True
gen_params['results_dir'] = './results/'
gen_params['num_colors'] = 15
gen_params['max_samples'] = 5000
gen_params['l2_norm_scale'] = 2
gen_params['ml_dataset'] = 'resistance_z'
gen_params['pickle_dir'] = '/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark_orig/parsed/02_08_19/1031114533/02082019analytetrials/pickled/'
gen_params['sklearn_random_state'] = 37
gen_params['ml_pca_n_components'] = 10
#[2,3,10,15,25]
gen_params['x0'] = None
gen_params['x1'] = [x1 for x1 in range(620,1020,20)]

####################################
# Kalman Filter Parameters
####################################

kf_params['n_signals'] = 20
kf_params['n_samples'] = 100
kf_params['sample_freq'] = 20.0
kf_params['h'] = 1.0
kf_params['q'] = list(np.logspace(-8,1,10))
kf_params['r'] = list(np.linspace(0.1,10,10))
                      
####################################
# Determine scaler and vector parameters
####################################

config_params_dicts = OrderedDict()

for dict_name, params_dict in params_dicts.items():

    # number of config files
    n_cfg_files = 1

    # keys for parameters that have more than one value
    vector_keys = []

    # determine vector keys and number of config files
    for k,v in params_dict.items():

        if isinstance(v, (list,)) and len(v) > 1:

            n_cfg_files *= len(v)
            vector_keys.append(k)

    # dictionary of scalar parameters
    scalar_params = { k:v for k,v in params_dict.items() if k not in vector_keys}
    vector_params = { k:v for k,v in params_dict.items() if k in vector_keys}

    ######################################
    # Generate dictionaries for each test
    ######################################

    # list of test dictionaries
    test_dicts = []

    if vector_params:

        # find all enumerations of the vector parameters
        vector_params_product = (dict(zip(vector_params, x)) for x in itertools.product(*vector_params.values()))

        # create separate dictionary for each vector parameter value
        for d in vector_params_product:

            test_dicts.append(d)
            test_dicts[-1].update(scalar_params)

    else:

        test_dicts.append(scalar_params)

    config_params_dicts[dict_name] = test_dicts

#######################
# Write Config Files
#######################

for experiment in experiments:

    #experiment_name, kf_dims, test_dataset_file = experiment
    experiment_name, kf_dims = experiment
    
    cfg_ctr = 1

    for config_params in itertools.product(config_params_dicts['gen'],
                                           config_params_dicts['kf']):


        #config_params[0]['load_path'] = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/evaluation/' + project + '/' + test_dataset_file
        config_params[1]['dimensions'] = kf_dims
        
        # Create Directories
        experiment_dir = '/Documents/wpi/research/code/dovebirdia/experiments/' + project + '/' + experiment_name + '/'
        model_dir_name = experiment_name + '_model_' + str(cfg_ctr) + '/'
        model_dir = os.environ['HOME'] + experiment_dir + model_dir_name
        results_dir = model_dir + '/results/'
        out_dir = model_dir + '/out'
        config_dir = model_dir + '/config/'

        if not os.path.exists(results_dir): os.makedirs(results_dir)
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        if not os.path.exists(config_dir): os.makedirs(config_dir)

        # Write Config Files
        for name, config_param in zip(config_params_dicts.keys(), config_params):

            cfg_file_name = model_dir_name[:-1] +  '_' + name + '.cfg'

            with open(config_dir + cfg_file_name, 'wb') as handle:

                dill.dump(config_param, handle)

        out_file_name = model_dir_name[:-1] + '.out'
        res_file_name = model_dir_name[:-1]

        # bash-batch script
        if machine == 'pengy':

            batch_string_prefix = 'python3 '

        else:

            batch_string_prefix = 'sbatch -o ./out/' + out_file_name + ' '

        batch_str = batch_string_prefix + script + ' -c ./config/' + ' -r ./results//\n'
        batch_file_name = model_dir + 'test_model.sh'
        batch_file = open(batch_file_name, 'w')
        batch_file.write(batch_str)
        batch_file.close()

        cfg_ctr += 1
