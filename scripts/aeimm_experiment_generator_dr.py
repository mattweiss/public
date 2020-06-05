#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=2G
#SBATCH -p short
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:0

import os, sys, socket
import numpy as np
np_float_prec = np.float64
import itertools
import tensorflow as tf
import dill
import itertools
from collections import OrderedDict
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.autoencoder import AutoencoderInteractingMultipleModel
from dovebirdia.filtering.kalman_filter import KalmanFilter
from dovebirdia.filtering.interacting_multiple_model import InteractingMultipleModel
from dovebirdia.deeplearning.regularizers.base import orthonormal_regularizer
from dovebirdia.deeplearning.activations.base import sineline, psineline, tanhpoly
import dovebirdia.utilities.dr_functions as drfns
import dovebirdia.math.distributions as distributions

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/dl_model.py'
project = 'imm'
experiment_name = 'aeimm_legendre_gaussian_F1_{F1}_Q1_{Q1}_F2_{F2}_Q2_{Q2}'.format(F1='NCV',Q1='1e-4',
                                                                                   F2='NCA',Q2='1e-4')
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

meta_params['network'] = AutoencoderInteractingMultipleModel

####################################
# Regularly edited Parameters
####################################

model_params['hidden_dims'] = [(128,64,32),(128,64),(64,32,16),(64,32)]
model_params['learning_rate'] = list(np.logspace(-3,-5,12))
model_params['optimizer'] = tf.train.AdamOptimizer
model_params['mbsize'] = 500

# model params

model_params['kf_type'] = InteractingMultipleModel
model_params['results_dir'] = '/results/'
model_params['input_dim'] = 2
model_params['output_dim'] = model_params['input_dim']

model_params['output_activation'] = None
model_params['activation'] = tf.nn.leaky_relu
model_params['use_bias'] = True
model_params['weight_initializer'] = tf.initializers.glorot_normal
model_params['bias_initializer'] = tf.initializers.zeros
model_params['weight_regularizer'] = None #[tf.keras.regularizers.l1,tf.keras.regularizers.l2]
model_params['weight_regularizer_scale'] = 0.0 #[1e-4,1e-5]
model_params['bias_regularizer'] = None
model_params['activity_regularizer'] = None
model_params['weight_constraint'] = None
model_params['bias_constraint'] = None
model_params['input_dropout_rate'] = 0.0
model_params['dropout_rate'] = 0.0
model_params['R_model'] = 'learned' # learned, identity
model_params['R_activation'] = None
model_params['train_ground'] = True

# loss
model_params['loss'] = tf.losses.mean_squared_error

# training
model_params['epochs'] = 10000
model_params['momentum'] = 0.96
model_params['use_nesterov'] = True
model_params['decay_steps'] = 100
model_params['decay_rate'] = 0.96
model_params['staircase'] = False

####################################
# Domain Randomization Parameters
####################################

ds_params['ds_type'] = 'train'
ds_params['x_range'] = (-1,1)

# set dt here based on x range and mb size, for use in scaling noise and the Kalman Filter
dt = (ds_params['x_range'][1]-ds_params['x_range'][0])/model_params['mbsize']

ds_params['n_trials'] = 1
ds_params['n_baseline_samples'] = 0
ds_params['n_samples'] = model_params['mbsize']
ds_params['n_features'] = model_params['input_dim']
ds_params['n_noise_features'] = ds_params['n_features']
ds_params['standardize'] = False
ds_params['feature_range'] = None
ds_params['baseline_shift'] = None
ds_params['param_range'] = 1.0
ds_params['max_N'] = 3
ds_params['min_N'] = 1
ds_params['metric_sublen'] = model_params['epochs'] // 100 # 1 percent
ds_params['fns'] = (
    #['zeros', drfns.zeros, []],
    #['exponential', drfns.exponential, [1.0,(0.02,0.045),-1.0]],
    #['sigmoid', drfns.sigmoid, [(0.0,100.0),0.15,60.0]],
    #['sine', drfns.sine, [(0,10.0),(0.01,0.01)]],
    ['taylor_poly', drfns.taylor_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(ds_params['max_N']+1)],
    #['legendre_poly', drfns.legendre_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(ds_params['max_N']+1)],
    #['trig_poly', drfns.trig_poly, [(-ds_params['param_range'],ds_params['param_range'])]*(2*ds_params['max_N']+1)],
)

ds_params['noise'] = [
    #[None, None, None],

    ['gaussian', np.random.multivariate_normal, {'mean':np.zeros(ds_params['n_features']),
                                                 'cov':dt*np.eye(ds_params['n_features'])}],

    # ['bimodal', distributions.bimodal, {'mean1':np.full(ds_params['n_features'],0.25),
    #                                     'cov1':0.02*np.eye(ds_params['n_features']),
    #                                     'mean2':np.full(ds_params['n_features'],-0.25),
    #                                     'cov2':0.02*np.eye(ds_params['n_features'])}],

    #['cauchy', np.random.standard_cauchy, {}],

    #['stable', distributions.stable, {'alpha':(1.0),'scale':(0.0)}], # alpha = 2 Gaussian, alpha = 1 Cauchy
]

####################################
# Kalman Filter Parameters
####################################

kf_params['with_z_dot'] = with_z_dot = False

#  measurements dimensions
kf_params['meas_dims'] = meas_dims = 8

#  state space dimensions
kf_params['state_dims'] = state_dims = kf_params['meas_dims']

# number of state estimate 
kf_params['dt'] = dt

# dynamical model order (i.e. ncv = 1, nca = 2, jerk = 3)
kf_params['model_order'] = model_order = 3

kf_params['H'] = np.kron(np.eye(meas_dims), np.eye(model_order+1)) if with_z_dot else np.kron(np.eye(meas_dims), np.array([1.0,0.0,0.0,0.0]))

# state-transition model

F_NCV = np.zeros((model_order+1,model_order+1))
F_NCA = np.zeros((model_order+1,model_order+1))
F = np.array([[1.0,dt,0.5*dt**2,(1.0/6.0)*dt**3],
              [0.0,1.0,dt,0.5*dt**2],
              [0.0,0.0,1.0,dt],
              [0.0,0.0,0.0,1.0]])

F_NCV[:F[np.ix_([0,1],[0,1])].shape[0],:F[np.ix_([0,1],[0,1])].shape[0] ] = F[np.ix_([0,1],[0,1])]
F_NCA[:F[np.ix_([0,1,2],[0,1,2])].shape[0],:F[np.ix_([0,1,2],[0,1,2])].shape[0] ] = F[np.ix_([0,1,2],[0,1,2])]
F_JERK = F

# process covariance

Q_NCV = np.zeros((model_order+1,model_order+1))
Q_NCA = np.zeros((model_order+1,model_order+1))
Q = np.eye(model_order+1)

Q_NCV[:Q[np.ix_([0,1],[0,1])].shape[0],:Q[np.ix_([0,1],[0,1])].shape[0] ] = Q[np.ix_([0,1],[0,1])]
Q_NCA[:Q[np.ix_([0,1,2],[0,1,2])].shape[0],:Q[np.ix_([0,1,2],[0,1,2])].shape[0] ] = Q[np.ix_([0,1,2],[0,1,2])]
Q_JERK = Q

# dictionary of models

kf_params['models'] = {
    'NCV1':[np.kron(np.eye(state_dims),F_NCV),1e-4*np.kron(np.eye(state_dims),Q_NCV)],
    #'NCV2':[np.kron(np.eye(state_dims),F_NCV),1e-4*np.kron(np.eye(state_dims),Q_NCV)]
    'NCA1':[np.kron(np.eye(state_dims),F_NCA),1e-4*np.kron(np.eye(state_dims),Q_NCA)],
    #'JERK1':[np.kron(np.eye(state_dims),F_JERK),1e-4*np.kron(np.eye(state_dims),Q_JERK)],
    #'NCV3':[np.kron(np.eye(state_dims),F_NCV),1e-4*np.kron(np.eye(state_dims),Q_NCV)],
    #'JERK1':[np.kron(np.eye(state_dims),F_JERK),1e-4*np.kron(np.eye(state_dims),Q_JERK)],
    # 'NCV2':[np.kron(np.eye(state_dims),F_NCV),1e-8*np.kron(np.eye(state_dims),Q_NCV)],
    # 'NCA2':[np.kron(np.eye(state_dims),F_NCA),1e-8*np.kron(np.eye(state_dims),Q_NCA)],
    # 'JERK2':[np.kron(np.eye(state_dims),F_JERK),1e-8*np.kron(np.eye(state_dims),Q_JERK)],
}
n_models = len(kf_params['models'].keys())

kf_params['R'] = None

####################
# Mixing Parameters
####################

p_trans = 0.1

kf_params['p'] = np.full((n_models,n_models),p_trans)
np.fill_diagonal(kf_params['p'],1.0-(n_models-1)*p_trans)
                                  
kf_params['mu'] = np.full((n_models,1),0.5)
st()
########################################
# Determine scaler and vector parameters
########################################

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