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
from dovebirdia.deeplearning.networks.lstm import LSTM
from keras import optimizers, losses, regularizers, constraints

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/weather_model.py'
project = 'weather'
experiment_name = 'lstm_weather_EPOCH_500_SAMPLES_40k_ALL_FEATURES'
experiment_dir = '/Documents/wpi/research/code/dovebirdia/experiments/' + project + '/' + experiment_name + '/'
machine = socket.gethostname()
####################################

meta_params = dict()
ds_params = dict()
model_params = dict()

params_dicts = OrderedDict([
    ('meta',meta_params),
    ('model',model_params),
    ('ds',ds_params),
])

####################################
# Meta Parameters
####################################

meta_params['network'] = LSTM

####################################
# Important Parameters
####################################

model_params['hidden_dims'] = [(512,128,64),(128,64),(128,64,32),(512,128)]
model_params['learning_rate'] = list(np.logspace(-3,-6,4))
model_params['seq_len'] = [1,5,15]
model_params['optimizer'] = optimizers.Adam
model_params['mbsize'] = [256,512]
model_params['weight_regularizer'] = None#[regularizers.l1,regularizers.l2]
model_params['weight_regularizer_scale'] = 0.0#[1e-4,1e-5]#list(np.logspace(-4,-8,5))

# model parameters

model_params['results_dir'] = '/results/'
model_params['input_dim'] = 5

model_params['output_dim'] = model_params['input_dim']
model_params['activation'] = tf.nn.leaky_relu
model_params['output_activation'] = None
model_params['use_bias'] = True
model_params['weight_initializer'] = 'glorot_normal'
model_params['bias_initializer'] = 1.0

model_params['bias_regularizer'] = None
model_params['activity_regularizer'] = None
model_params['bias_constraint'] = None
model_params['recurrent_regularizer'] = None
model_params['stateful'] = False
model_params['return_seq'] = True
model_params['input_dropout_rate'] = 0.0
model_params['dropout_rate'] = 0.0
model_params['weight_constraint'] = None #constraints.MaxNorm(1.0)

# loss
model_params['loss'] = losses.mean_squared_error

# training
model_params['epochs'] = 500
model_params['momentum'] = 0.96
model_params['use_nesterov'] = True
model_params['decay_steps'] = 1000
model_params['decay_rate'] = 1.0
model_params['staircase'] = False
model_params['outliers'] = False
model_params['p_outlier'] = 0.05 * (2.0/3.0)
model_params['outlier_range'] = (0,1)

####################################
# Dataset Parameters
####################################

ds_params['saved_dataset'] = '/home/mlweiss/Documents/wpi/research/data/weather/historicalHourlyWeatherData/split/hourly_weather_dataset_FEATURES_temperature_pressure_humidity_wind_direction_wind_speed_CITY_Boston_SAMPLES_40000.pkl'

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
                                       config_params_dicts['ds']):

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
