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
import pickle
import itertools
from collections import OrderedDict
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.autoencoder import AutoencoderKalmanFilter

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/train_model.py'
test_name = 'aekf_KILLME2'
test_dir = '/Documents/wpi/research/code/dovebirdia/models/' + test_name + '/'
machine = socket.gethostname()
####################################

meta_params = dict()
dr_params = dict()
kf_params = dict()
model_params = dict()

params_dicts = OrderedDict([
    ('meta',meta_params),
    ('model',model_params),
    ('dr',dr_params),
    ('kf',kf_params),
])

####################################
# Meta Parameters
####################################

meta_params['model'] = AutoencoderKalmanFilter
#meta_params['fit'] = AutoencoderKalmanFilter.fitDomainRandomization

####################################
# Model Parameters
####################################

model_params['results_dir'] = './saved_weights/'
model_params['input_dim'] = 1
model_params['output_dim'] = 1
model_params['hidden_dims'] = (256,64) # if using AEKF append number of signals from KF to hidden_dims in train_model.py, otherwise include here
model_params['output_activation'] = None
model_params['activation'] = tf.nn.leaky_relu
model_params['use_bias'] = True
model_params['kernel_initializer'] = 'glorot_uniform'
model_params['bias_initializer'] = 'zeros'
model_params['kernel_regularizer'] = None
model_params['bias_regularizer'] = None
model_params['activity_regularizer'] = None
model_params['kernel_constraint'] = None
model_params['bias_constraint'] = None

# loss
model_params['loss'] = tf.losses.mean_squared_error

# training
model_params['epochs'] = 1000
model_params['mbsize'] = 100
model_params['optimizer'] = tf.train.AdamOptimizer
model_params['learning_rate'] = 1e-3
#list(np.logspace(-3,-5,10))

# metric(s)
model_params['metrics'] = None
model_params['test_size'] = 10

####################################
# Domain Randomization Parameters
####################################

def exponential_fn(x,a,b,c):

    return a * np.exp(b*x) + c

def sigmoid_fn(x,a,b,c):

    y = a * (1 + np.exp(-b * (x - c)))**-1
    y -= y[0]
    return y

dr_params['x_range'] = (0,100)
dr_params['n_samples'] = 100
dr_params['fns'] = [['sigmoid', sigmoid_fn, [(0,100),0.15,60.0]]]
dr_params['noise'] = np.random.normal
dr_params['noise_params'] = {'loc':0.0, 'scale':5.0}

####################################
# Kalman Filter Parameters
####################################

kf_params['dimensions'] = (1,2)
#kf_params['model'] = 'ncv'
kf_params['n_signals'] = 1
kf_params['n_samples'] = 100
kf_params['sample_freq'] = 1.0
kf_params['h'] = (1.0,0.0)
kf_params['q'] = 1e-4
#list(np.logspace(-8,1,10))

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

cfg_ctr = 1

for config_params in itertools.product(config_params_dicts['meta'],
                                       config_params_dicts['model'],
                                       config_params_dicts['dr'],
                                       config_params_dicts['kf']):

    # Create Directories
    subtest_dir_name = test_name + '_test_' + str(cfg_ctr) + '/'
    subtest_dir = os.environ['HOME'] + test_dir + subtest_dir_name
    results_dir = subtest_dir + '/results/'
    out_dir = subtest_dir + '/out'
    config_dir = subtest_dir + '/config/'
    
    if not os.path.exists(results_dir): os.makedirs(results_dir)
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    if not os.path.exists(config_dir): os.makedirs(config_dir)

    # Write Config Files
    for name, config_param in zip(config_params_dicts.keys(), config_params):

        cfg_file_name = subtest_dir_name[:-1] +  '_' + name + '.cfg'

        with open(config_dir + cfg_file_name, 'wb') as handle:

            dill.dump(config_param, handle)
        
    out_file_name = subtest_dir_name[:-1] + '.out'
    res_file_name = subtest_dir_name[:-1]

    # bash-batch script
    if machine == 'pengy':

        batch_string_prefix = 'python3 '

    else:

        batch_string_prefix = 'sbatch -o ./out/' + out_file_name + ' '
        
    batch_str = batch_string_prefix + script + ' -c ./config/' + ' -r ./results//\n'
    batch_file_name = subtest_dir + 'run.sh'
    batch_file = open(batch_file_name, 'w')
    batch_file.write(batch_str)
    batch_file.close()
    
    cfg_ctr += 1
