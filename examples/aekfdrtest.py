import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from dovebirdia.deeplearning.networks.autoencoder import AutoencoderKalmanFilter

# define domain randomization functions
# dr_fns is a dictionary with a single key whose value is a list.
# Each element of this list is a list defining a function: [ function name, function definition, parameters ]
# if parameters is a tuple that is the range from which the parameter is drawn

def exponential_fn(x,a,b,c):

    return a * np.exp(b*x) + c

def sigmoid_fn(x,a,b,c):

    y = a * (1 + np.exp(-b * (x - c)))**-1
    y -= y[0]
    return y

dr_params = dict()
dr_params['x_range'] = (0,100)
dr_params['n_samples'] = 100
dr_params['fns'] = [
    #['exponential', exponential_fn, [1.0,(0.02,0.045),-1.0]],
    ['sigmoid', sigmoid_fn, [(0.0,100.0),0.15,60.0]],
]
dr_params['noise'] = np.random.normal
dr_params['noise_params'] = {'loc':0.0, 'scale':5.0}

# parameters dictionary
model_params = dict()

# output params
model_params['model_name'] = 'aedrtest'
model_params['results_dir'] = './saved_weights/'

# network params
model_params['input_dim'] = 1
model_params['output_dim'] = 1
model_params['hidden_dims'] = [ 256,64,4 ]
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
model_params['epochs'] = 5
model_params['mbsize'] = 100
model_params['optimizer'] = tf.train.AdamOptimizer
model_params['learning_rate'] = 1e-3
model_params['test_size'] = 100

# KF params
kf_params = dict()
kf_params['dimensions'] = (1,2)
#kf_params['model'] = 'ncv'
kf_params['n_signals'] = model_params['hidden_dims'][-1]
kf_params['n_samples'] = 100
kf_params['sample_freq'] = 1.0
kf_params['h'] = [1.0,0.0]
kf_params['q'] = 1e-4

# Network
nn = AutoencoderKalmanFilter(model_params, kf_params)
print(nn.__class__)
nn.getModelSummary()
nn.fitDomainRandomization(dr_params, save_weights=False)
