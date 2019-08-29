import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from dovebirdia.deeplearning.networks.base import AbstractNetwork
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.networks.autoencoder import Autoencoder

# define domain randomization functions
# dr_fns is a dictionary with a single key whose value is a list.
# Each element of this list is a list defining a function: [ function name, function definition, parameters ]
# if parameters is a tuple that is the range from which the parameter is drawn
dr_params = dict()
dr_params['x_range'] = (0,100)
dr_params['n_samples'] = 100
dr_params['fns'] = [['exponential', lambda x,a,b,c : a * np.exp(b*x) + c, [1.0,(0.04605170185988092, 0.02302585092994046),-1.0]]]
dr_params['noise'] = np.random.normal
dr_params['noise_params'] = {'loc':0.0, 'scale':5.0}

# parameters dictionary
params = dict()

# output params
params['model_name'] = 'aedrtest'
params['results_dir'] = './saved_weights/'

# network params
params['input_dim'] = 1
params['output_dim'] = 1
params['hidden_dims'] = [ 256,64,32,1 ]
params['output_activation'] = None
params['activation'] = tf.nn.relu
params['use_bias'] = True
params['kernel_initializer'] = 'glorot_normal'
params['bias_initializer'] = 'zeros'
params['kernel_regularizer'] = None
params['bias_regularizer'] = None
params['activity_regularizer'] = None
params['kernel_constraint'] = None
params['bias_constraint'] = None

# loss
params['loss'] = tf.keras.losses.MeanSquaredError()

# training
params['epochs'] = 1000
params['mbsize'] = 100
params['optimizer'] = tf.train.AdamOptimizer
params['learning_rate'] = 1e-3

# Network
nn = Autoencoder(params)
print(nn.__class__)
nn.getModelSummary()
nn.fitDomainRandomization(dr_params, save_weights=False)
