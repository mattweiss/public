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
# from dovebirdia.deeplearning.regularizers.base import orthonormal_regularizer
from dovebirdia.deeplearning.activations.base import sineline, psineline, tanhpoly
import dovebirdia.utilities.dr_functions as drfns
import dovebirdia.stats.distributions as distributions

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/weather_model.py'
project = 'weather'
experiment_name = 'aekf_KILLME'
experiment_dir = '/Documents/wpi/research/code/dovebirdia/experiments/' + project + '/' + experiment_name + '/'
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

meta_params['network'] = AutoencoderKalmanFilter

####################################
# Model Parameters
####################################

model_params['results_dir'] = '/results/'
model_params['input_dim'] = 5
model_params['output_dim'] = model_params['input_dim']
model_params['hidden_dims'] = [(128,64),(128,64,32),(64,32)]
model_params['output_activation'] = None
model_params['activation'] = tf.nn.leaky_relu
model_params['use_bias'] = True
model_params['weight_initializer'] = tf.initializers.glorot_uniform
model_params['bias_initializer'] = tf.initializers.zeros
model_params['weight_regularizer'] = tf.keras.regularizers.l2
model_params['weight_regularizer_scale'] = 0.0
model_params['bias_regularizer'] = None
model_params['activity_regularizer'] = None
model_params['weight_constraint'] = None
model_params['bias_constraint'] = None
model_params['input_dropout_rate'] = 0.0
model_params['dropout_rate'] = 0.0
model_params['R_model'] = 'learned' # learned, identity
model_params['R_activation'] = None

# loss
model_params['loss'] = tf.losses.mean_squared_error

# training
model_params['epochs'] = 100
model_params['mbsize'] = 960
model_params['optimizer'] = tf.train.AdamOptimizer
model_params['momentum'] = 0.95
model_params['learning_rate'] = list(np.logspace(-3,-5,3))
model_params['trials'] = [0]

####################################
# Dataset Parameters
####################################

ds_params['saved_dataset'] = '/home/mlweiss/Documents/wpi/research/data/weather/split/weather_all_train_test_split.npy'

####################################
# Kalman Filter Parameters
####################################

kf_params['dimensions'] = (1,2)
kf_params['n_signals'] = 16
kf_params['n_measurements'] = model_params['mbsize']
kf_params['dt'] = 0.2
kf_params['sample_freq'] = kf_params['dt']**-1
kf_params['q'] = list(np.logspace(-2,0,5)) # list(np.logspace(-1,-8,8))
kf_params['f_model'] = 'fixed' # fixed, random, learned
kf_params['h_model'] = 'fixed' # fixed, random, learned, identity
kf_params['weight_initializer'] = tf.initializers.glorot_uniform # if learning F and H

# Build dynamical model
if kf_params['dimensions'][1] == 2:

    # F
    if kf_params['f_model'] == 'fixed':

        kf_params['F'] = np.kron(np.eye(kf_params['n_signals']), np.array([[1.0,kf_params['dt']],[0.0,1.0]]))

    elif kf_params['f_model'] == 'random':

        kf_params['F'] = np.random.normal(size=(kf_params['n_signals']*kf_params['dimensions'][1],kf_params['n_signals']*kf_params['dimensions'][1]))

    # H
    if kf_params['h_model'] == 'fixed':

        kf_params['H'] = np.kron(np.eye(kf_params['n_signals']), np.array([1.0,0.0]))

    elif kf_params['h_model'] == 'identity':

        kf_params['H'] = np.kron(np.eye(kf_params['n_signals']), np.array([1.0,1.0]))

    elif kf_params['h_model'] == 'random':

        kf_params['H'] = np.random.normal(size=(kf_params['n_signals'],kf_params['n_signals']*kf_params['dimensions'][1]))

if kf_params['dimensions'][1] == 3:

    # F
    if kf_params['f_model'] == 'fixed':

        kf_params['F'] = np.kron(np.eye(kf_params['n_signals']), np.array([[1.0,kf_params['dt'],0.5*kf_params['dt']**2],[0.0,1.0,kf_params['dt']],[0.0,0.0,1.0]]))

    elif kf_params['f_model'] == 'random':

        kf_params['F'] = np.random.normal(size=(kf_params['n_signals']*kf_params['dimensions'][1],kf_params['n_signals']*kf_params['dimensions'][1]))

    # H
    if kf_params['h_model'] == 'fixed':

        kf_params['H'] = np.kron(np.eye(kf_params['n_signals']), np.array([1.0,0.0,0.0]))

    elif kf_params['h_model'] == 'random':

        kf_params['H'] = np.random.normal(size=(kf_params['n_signals'],kf_params['n_signals']*kf_params['dimensions'][1]))

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
                                       config_params_dicts['ds'],
                                       config_params_dicts['kf']):

    # Create Directories
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
    batch_file_name = model_dir + 'train_model.sh'
    batch_file = open(batch_file_name, 'w')
    batch_file.write(batch_str)
    batch_file.close()

    cfg_ctr += 1
