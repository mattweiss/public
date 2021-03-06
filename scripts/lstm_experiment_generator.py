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
from keras import optimizers, losses
import dill
import itertools
from collections import OrderedDict
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.lstm_tf import LSTM
import dovebirdia.utilities.dr_functions as drfns
import dovebirdia.stats.distributions as distributions

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/dl_model.py'
project = 'pets'
experiment_name = 'lstm_legendre_mask_percent_1_value_1000_KILLME'
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

model_params['hidden_dims'] = (128,64) # [(512,128,64),(128,64),(128,64,32),(512,128)]
model_params['learning_rate'] = 1e-3 # list(np.logspace(-3,-5,3))
model_params['seq_len'] = 1 # [1,5,10,15]
model_params['optimizer'] = optimizers.Adam
model_params['mbsize'] = [500,125]

# model parameters

model_params['results_dir'] = '/results/'
model_params['input_dim'] = 2
model_params['output_dim'] = model_params['input_dim']
model_params['output_activation'] = None
model_params['activation'] = tf.nn.leaky_relu
model_params['use_bias'] = True
model_params['weight_initializer'] = 'glorot_normal'
model_params['bias_initializer'] = 1.0
model_params['weight_regularizer'] = None
model_params['weight_regularizer_scale'] = 0.0
model_params['bias_regularizer'] = None
model_params['activity_regularizer'] = None
model_params['weight_constraint'] = None
model_params['bias_constraint'] = None
model_params['recurrent_regularizer'] = None
model_params['input_dropout_rate'] = 0.0
model_params['dropout_rate'] = 0.0
model_params['stateful'] = False
model_params['return_seq'] = True
model_params['train_ground'] = False

# loss
model_params['loss'] = tf.losses.mean_squared_error # losses.mean_squared_error

# training
model_params['epochs'] = 10

# testing
# model_params['history_size'] = int(0.01*model_params['epochs'])

####################################
# Domain Randomization Parameters
####################################

ds_params['ds_type'] = 'train'
ds_params['x_range'] = (-1,1)
ds_params['n_trials'] = 1
ds_params['n_baseline_samples'] = 0
ds_params['n_samples'] = 500
ds_params['n_features'] = model_params['input_dim']
ds_params['n_noise_features'] = model_params['input_dim']
ds_params['feature_range'] = None
ds_params['baseline_shift'] = None
ds_params['param_range'] = 1.0
ds_params['max_N'] = 15
ds_params['min_N'] = 3
ds_params['mask_percent'] = 0.01
ds_params['mask_value'] = 1000

ds_params['fns'] = (
    # ['exponential', drfns.exponential, [1.0,(0.02,0.045),-1.0]],
    #['sigmoid', drfns.sigmoid, [(0.0,100.0),0.15,60.0]],
    #['sine', drfns.sine, [(0.0,100.0),(0.04,0.1)]],
    #['taylor_poly', drfns.taylor_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(ds_params['max_N']+1)],
    ['legendre_poly', drfns.legendre_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(ds_params['max_N']+1)],
    #['trig_poly', drfns.trig_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(2*ds_params['max_N']+1)],
)

ds_params['noise'] = [
    ['gaussian', np.random.normal, {'loc':0.0, 'scale':0.0}],
    # ['bimodal', distributions.bimodal, {'loc1':0.05, 'scale1':0.1, 'loc2':-0.05, 'scale2':0.1}],
    # ['cauchy', np.random.standard_cauchy, {}],
    # ['stable', distributions.stable, {'alpha':(1.0),'scale':0.2}],
    #['stable', distributions.stable, {'alpha':(1.5), 'scale':0.01}],
]

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
