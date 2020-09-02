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
from dovebirdia.filtering.kalman_filter import ExtendedKalmanFilter
import dovebirdia.utilities.dr_functions as drfns 
from dovebirdia.datasets.domain_randomization import DomainRandomizationDataset
from sklearn.datasets import make_spd_matrix

#################################
# Function to generate SPD Matrix
#################################
def generate_spd(ndim=2,scale=1,epsilon=1e-8):

    m = scale * np.random.normal(size=(ndim,ndim))
    L = np.abs(np.triu(m))
    return L.T@L

####################################
# Test Name and Description
####################################
script = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/scripts/filter_model.py'
#****************************************************************************************************************************
project = 'asilomar2020'

experiments = [
    ('ekfct_turn_1_gaussian_0_20_Q_0-5',
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

meta_params['filter'] = ExtendedKalmanFilter

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
kf_params['state_dims'] = state_dims = 5

# number of state estimate 
kf_params['dt'] = dt = 1e-1

# dynamical model order (i.e. ncv = 2, nca = 3, jerk = 4)
kf_params['model_order'] = model_order = 1

#########
# Models
#########

def F(dt,x):

    w = x[4,0]
    wt = w*dt

    # if w is zero
    f0 = np.array([
        [1.0,dt,0.0,0.0,0.0],
        [0.0,1.0,0.0,0.0,0.0],
        [0.0,0.0,1.0,dt,0.0],
        [0.0,0.0,0.0,1.0,0.0],
        [0.0,0.0,0.0,0.0,0.0]
    ])

    # if w is non-zero
    f1 = tf.TensorArray(tf.float64, size=5, dynamic_size=True)
    f1 = f1.write(0,[1.0,tf.sin(wt)/w,0.0,-(1-tf.cos(wt))/w,0.0])
    f1 = f1.write(1,[0.0,tf.cos(wt),0.0,-tf.sin(wt),0.0])
    f1 = f1.write(2,[0.0,(1-tf.cos(wt))/w,1.0,tf.sin(wt)/w,0.0])
    f1 = f1.write(3,[0.0,tf.sin(wt),0.0,tf.cos(wt),0.0])
    f1 = f1.write(4,[0.0,0.0,0.0,0.0,1.0])
    f1 = tf.reshape(f1.stack(),(5,5))

    return tf.cond(tf.equal(w,0.0),lambda:f0,lambda:f1)
    #return tf.cond(tf.less(tf.abs(w),1e-8),lambda:f0,lambda:f1)

def J(dt,x):

    x_dot,y_dot = x[1,0], x[3,0]

    w = x[4,0]
    wt = w*dt

    # if w is zero
    j0 = tf.TensorArray(tf.float64, size=5, dynamic_size=True)
    j0 = j0.write(0,[1.0,dt,0.0,0.0,-0.5*(dt**2)*y_dot])
    j0 = j0.write(1,[0.0,1.0,0.0,0.0,-dt*y_dot])
    j0 = j0.write(2,[0.0,0.0,1.0,dt,0.5*(dt**2)*x_dot])
    j0 = j0.write(3,[0.0,0.0,0.0,1.0,dt*x_dot])
    j0 = j0.write(4,[0.0,0.0,0.0,0.0,1.0])
    j0 = tf.reshape(j0.stack(),(5,5))

    # if w is non-zero
    f1_w = x_dot*( (tf.cos(wt)*dt)/w - tf.sin(wt)/w**2) - y_dot*( (tf.sin(wt)*dt)/w - (1-tf.cos(wt))/w**2)
    f2_w = -x_dot*tf.sin(wt)*dt - y_dot*tf.cos(wt)*dt
    f3_w = x_dot*( (tf.sin(wt)*dt)/w - (1-tf.cos(wt))/w**2) + y_dot*( (tf.cos(wt)*dt)/w - (tf.sin(wt)*dt)/w**2)
    f4_w = x_dot*tf.cos(wt)*dt - y_dot*tf.sin(wt)*dt

    j1 = tf.TensorArray(tf.float64, size=5, dynamic_size=True)
    j1 = j1.write(0,[1.0, tf.sin(wt)/w, 0.0, (-1+tf.cos(wt))/w, f1_w])
    j1 = j1.write(1,[0.0, tf.cos(wt), 0.0, -tf.sin(wt), f2_w])
    j1 = j1.write(2,[0.0, (1-tf.cos(wt))/w, 1.0, tf.sin(wt)/w, f3_w])
    j1 = j1.write(3,[0.0, tf.sin(wt), 0.0, tf.cos(wt), f4_w])
    j1 = j1.write(4,[0.0,0.0,0.0,0.0,1.0])
    j1 = tf.reshape(j1.stack(),(5,5))
    
    return tf.cond(tf.equal(w,0.0),lambda:j0,lambda:j1)
    #return tf.cond(tf.less(tf.abs(w),1e-8),lambda:j0,lambda:j1)

def Q(dt):

    G = np.array([
        [(dt**2)/2.0,0.0,0.0],
        [dt,0.0,0.0],
        [0.0,(dt**2)/2.0,0.0],
        [0.0,dt,0.0],
        [0.0,0.0,dt]
    ])

    w = np.array([[0.25],[0.25],[0.1]])
    Q = 1e-2*w@w.T

    return G@Q@G.T

kf_params['F'] = F
kf_params['F_params'] = ('dt')

kf_params['J'] = J
kf_params['J_params'] = ('dt')

kf_params['Q'] = Q
kf_params['Q_params'] = ('dt')

kf_params['H'] = np.array([
    [1.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,1.0,0.0,0.0]
    ])

kf_params['R'] = 20.0*np.eye(meas_dims)

#####################
# AEKF MCA Parameters
#####################

# diagonal
# kf_params['R'] = 1.0 * np.eye(meas_dims)
# kf_params['Q'] = 1e-4*np.eye((model_order+1)*state_dims)

# logspace diagonal
#kf_params['R'] = [ r*np.eye(meas_dims) for r in np.logspace(2,-8,10) ]
# kf_params['R'] = None # [ None for r in np.logspace(2,-8,10) ]
# kf_params['Q'] = 1e-2 #[ q*np.eye((model_order+1)*state_dims) for q in np.logspace(-2,-8,4) ]

# random spd
# kf_params['R'] = [ generate_spd(meas_dims,scale=10) for _ in np.arange(10) ]
# kf_params['Q'] = [ generate_spd((model_order+1)*state_dims,scale=10) for _ in np.arange(10) ]

####################################
# Determine scaler and vector parameters
####################################

config_params_dicts = OrderedDict()

for dict_name, params_dict in params_dicts.items():

    # number of config files+
    
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
