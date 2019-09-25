import numpy as np
import tensorflow as tf
from pdb import set_trace as st

from dovebirdia.deeplearning.networks.autoencoder import Autoencoder, AutoencoderKalmanFilter
from dovebirdia.datasets.mnist import MNISTDataset

# load MNIST
ds_params = {
    'with_val':True,
    'onehot':False,
    'supervised':False,
    }

mnist_dataset = MNISTDataset(ds_params).getDataset()

# parameters dictionary
params = dict()

# network params
params['input_dim'] = 784
params['output_dim'] = 784
params['hidden_dims'] = [ 128, 64, 1 ]
params['output_activation'] = tf.nn.sigmoid
params['activation'] = tf.nn.sigmoid
params['use_bias'] = True
params['weight_initializer'] = tf.initializers.glorot_uniform
params['bias_initializer'] = tf.initializers.zeros
params['weight_regularizer'] = None
params['bias_regularizer'] = None
params['activity_regularizer'] = None
params['weight_constraint'] = None
params['bias_constraint'] = None

# loss
params['loss'] = tf.losses.mean_squared_error

# training
params['epochs'] = 1
params['mbsize'] = 32
params['optimizer'] = tf.train.AdamOptimizer
params['learning_rate'] = 1e-3

######################
# Kalman Filter
######################

kf_params = dict()
kf_params['dimensions'] = (1,2)
#kf_params['model'] = 'ncv'
kf_params['n_signals'] = 1
kf_params['n_samples'] = 100
kf_params['sample_freq'] = 1.0
#kf_params['r'] = 100.0
#R = np.kron(np.eye( kf_params['n_signals']), np.array(kf_params['r'], dtype=np.float64))
kf_params['h'] = [1.0,0.0]
kf_params['q'] = 1e-2
kf_params['r'] = 1.0

# Network
nn = AutoencoderKalmanFilter(params, kf_params)
#nn = Autoencoder(params)
print(nn.__class__)
nn.getModelSummary()
nn.fit(mnist_dataset)
