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
from tensorflow import keras
import dill
import itertools
from collections import OrderedDict
from pdb import set_trace as st

from dovebirdia.deeplearning.networks.keras_classifiers import KerasMultiLabelClassifier

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/ccdc_model.py'
project = 'ccdc_mixtures'
experiment_name = '07_12_19_mean_T_std_F'
experiment_dir = '/Documents/wpi/research/code/dovebirdia/experiments/' + project + '/' + experiment_name + '/'
machine = socket.gethostname()
####################################

meta_params = dict()
dataset_params = dict()
model_params = dict()

params_dicts = OrderedDict([
    ('meta',meta_params),
    ('dataset',dataset_params),
    ('model',model_params),
])

####################################
# Meta Parameters
####################################

meta_params['network'] = KerasMultiLabelClassifier

####################################
# Dataset Parameters
####################################

dataset_params['dataset_dir'] = '/home/mlweiss/Documents/wpi/research/data/ccdc/dvd_dump_clark_3/split/07_12_19_mean_T_std_F/'
dataset_params['with_val'] = True
dataset_params['resistance_type'] = 'resistance_z' # ['resistance','resistance_kf0','resistance_kf1']
dataset_params['labels'] = None
dataset_params['sensors'] = None
dataset_params['with_synthetic'] = False
dataset_params['samples'] = [(0,600),(0,700),(0,800),(0,900),(0,1000)]
dataset_params['labels'] = 'concentration' #'binary_presence' # 'binary_presence' in {0,1}, 'concentration' in R^n
dataset_params['standardize'] = True

####################################
# Model Parameters
####################################

model_params['results_dir'] = '/results/'
model_params['output_dim'] = 5 # number of classes
model_params['hidden_dims'] = [(512,256,64,32),(256,64,32),(512,256,64)]
model_params['activation'] = tf.nn.leaky_relu
model_params['output_activation'] = tf.nn.relu
model_params['use_bias'] = True
model_params['kernel_initializer'] = 'glorot_normal'
model_params['bias_initializer'] = tf.initializers.ones # [tf.initializers.zeros,tf.initializers.ones]
model_params['kernel_regularizer'] = keras.regularizers.l2
model_params['kernel_regularizer_scale'] = 0.0
model_params['bias_regularizer'] = None
model_params['bias_regularizer_scale'] = None
model_params['activity_regularizer'] = None
model_params['activity_regularizer_scale'] = None
model_params['kernel_constraint'] = None
model_params['bias_constraint'] = None
model_params['dropout_rate'] = 0.0
model_params['input_dropout_rate'] = 0.0
model_params['scale_output'] = False
model_params['early_stopping'] = False

# loss
model_params['loss'] = tf.keras.losses.mean_squared_error
# tf.nn.sigmoid_cross_entropy_with_logits
# tf.keras.losses.mean_squared_error
# tf.keras.losses.categorical_crossentropy
model_params['from_logits'] = True
model_params['metrics'] = (['mse'])

# training
model_params['epochs'] = 1000
model_params['mbsize'] = [32,64]
model_params['optimizer'] = tf.train.AdamOptimizer
model_params['optimizer_params'] = [{'learning_rate':lr} for lr in np.logspace(-4,-6,3)]
#[ {'learning_rate':lr,'momentum':0.95,'use_nesterov':True} for lr in np.logspace(-4,-8,5) ]
#[{'learning_rate':lr} for lr in np.logspace(-4,-7,4)]

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
                                       config_params_dicts['dataset'],
                                       config_params_dicts['model'],
):

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
