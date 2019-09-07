import tensorflow as tf
from pdb import set_trace as st

from dovebirdia.deeplearning.networks.base import AbstractNetwork
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.networks.autoencoder import Autoencoder
from dovebirdia.datasets.mnist import MNISTDataset

# load MNIST
ds_params = {
    'with_val':True,
    'onehot':False,
    'supervised':False,
    }

mnist_dataset = MNISTDataset(params=ds_params).getDataset()

# parameters dictionary
params = dict()

# output params
params['model_name'] = 'aetest'
params['results_dir'] = './saved_weights/'

# network params
params['input_dim'] = 784
params['output_dim'] = 784
params['hidden_dims'] = [ 128, 64, 16 ]
params['output_activation'] = tf.nn.sigmoid
params['activation'] = tf.nn.sigmoid
params['use_bias'] = True
params['kernel_initializer'] = 'glorot_normal'
params['bias_initializer'] = 'zeros'
params['kernel_regularizer'] = None
params['bias_regularizer'] = None
params['activity_regularizer'] = None
params['kernel_constraint'] = None
params['bias_constraint'] = None

# loss
params['loss'] = tf.losses.mean_squared_error

# training
params['epochs'] = 3
params['mbsize'] = 32
params['optimizer'] = tf.train.AdamOptimizer
params['learning_rate'] = 1e-3

# metrics
#params['metrics'] = [tf.keras.metrics.MeanSquaredError()]

# Network
nn = Autoencoder(params)
print(nn.__class__)
nn.getModelSummary()
nn.fit(mnist_dataset, save_weights=True)
