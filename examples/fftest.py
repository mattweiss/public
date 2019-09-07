import tensorflow as tf
from pdb import set_trace as st

from dovebirdia.deeplearning.networks.base import AbstractNetwork
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.networks.autoencoder import Autoencoder
from dovebirdia.datasets.ccdc_mixtures import ccdcMixturesDataset

# load MNIST
ds_params = {
    'dataset_dir':'/home/mlweiss/Documents/wpi/research/code/sensors/mixtures/datasets/02_05_19-0905144322/',
    'val_size':0.1,
    'resistance_type':'resistance_z',
    'labels':None,
    'sensors':None,
    'with_synthetic':True,
    }

ccdc_dataset = ccdcMixturesDataset(params=ds_params).getDataset()

for k,v in ccdc_dataset.items():

    print(k,v.shape)
st()
# parameters dictionary
params = dict()

# network params
params['input_dim'] = 784
params['output_dim'] = 10
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
params['loss'] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# training
params['epochs'] = 25
params['mbsize'] = 32
params['optimizer'] = tf.train.AdamOptimizer
params['learning_rate'] = 1e-3

# metrics
#params['metrics'] = [tf.keras.metrics.MeanSquaredError()]

# Network
nn = FeedForwardNetwork(params)
print(nn.__class__)
nn.getModelSummary()
nn.fit(mnist_dataset)
