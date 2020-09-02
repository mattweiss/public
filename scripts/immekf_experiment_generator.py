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
from dovebirdia.filtering.interacting_multiple_model_kalman_filter import InteractingMultipleModelExtendedKalmanFilter
import dovebirdia.utilities.dr_functions as drfns 

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/filter_model.py'
#****************************************************************************************************************************
project = 'asilomar2020'

experiments = [
    ('immekf_ncv1_turn_1_gaussian_0_20_Q_0-5',
     '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/asilomar2020/eval/benchmark_gaussian_20_turn.pkl')
]

#****************************************************************************************************************************
machine = socket.gethostname()
####################################

meta_params = dict()
ds_params = dict()
kf_params = dict()
model_params = dict()

params_dicts = OrderedDict([
    ('meta',meta_params),
    ('model',model_params),
    ('ds',ds_params),
    ('kf',kf_params),
])

####################################
# Meta Parameters
####################################

meta_params['filter'] = InteractingMultipleModelExtendedKalmanFilter

####################################
# Model Parameters
####################################

model_params['results_dir'] = '/results/'

####################################
# Kalman Filter Parameters
####################################

#  measurements dimensions
kf_params['meas_dims'] = meas_dims = 2

#  state space dimensions
kf_params['state_dims'] = state_dims = kf_params['meas_dims']

# number of state estimate 
kf_params['dt'] = dt = 1e-1

# dynamical model order (i.e. ncv = 2, nca = 3, jerk = 4)
kf_params['model_order'] = model_order = 2

#########
# Models
#########

#NCV
# np.array([[1.0,dt],
#           [0.0,1.0]])

# # NCA
# np.array([[1.0,dt,0.5*dt**2],
#           [0.0,1.0,dt],
#           [0.0,0.0,1.0]])

# # Jerk
# np.array([[1.0,dt,0.5*dt**2,(1.0/6.0)*dt**3],
#           [0.0,1.0,dt,0.5*dt**2],
#           [0.0,0.0,1.0,dt],
#           [0.0,0.0,0.0,1.0]])

def F_NCV(state_dims,dt,x):

    state_trans = np.array([[1.0,dt],
                            [0.0,1.0]])
    
    return np.kron(np.eye(state_dims),state_trans)

def Q(state_dims,model_order):

    return 1e-4*np.kron(np.eye(state_dims),np.eye(model_order))

# dictionary of models

kf_params['models'] = {
    'NCV':[F_NCV,('state_dims','dt'), # F/F params
           F_NCV,('state_dims','dt'), # J/ J params
           Q,('state_dims','model_order')], # Q
}

n_models = len(kf_params['models'].keys())

kf_params['H'] = np.kron(np.eye(meas_dims), np.insert(np.zeros(model_order-1),0,1) )
kf_params['R'] = 20.0 * np.eye(meas_dims)

####################
# Mixing Parameters
####################

p_trans = 0.1

kf_params['p'] = np.full((n_models,n_models),p_trans)
np.fill_diagonal(kf_params['p'],1.0-(n_models-1)*p_trans)
                                  
kf_params['mu'] = np.full((n_models,1),0.5)

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

    experiment_name, test_dataset_full_path = experiment
    
    cfg_ctr = 1

    for config_params in itertools.product(config_params_dicts['meta'],
                                           config_params_dicts['model'],
                                           config_params_dicts['ds'],
                                           config_params_dicts['kf']):


        config_params[2]['load_path'] = test_dataset_full_path
        #config_params[3]['dimensions'] = kf_dims
        
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
