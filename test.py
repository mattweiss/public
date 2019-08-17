import tensorflow as tf

from deeplearning.abstract_network import AbstractNetwork
from deeplearning.feedforward_network import FeedForwardNetwork

# Load a toy dataset for the sake of this example
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

dataset = dict()
dataset['x_train'] = x_train
dataset['x_test'] = x_test
dataset['x_val'] = x_val
dataset['y_train'] = y_train
dataset['y_test'] = y_test
dataset['y_val'] = y_val


# parameters dictionary
params = dict()

# network params
params['input_dim'] = 784
params['output_dim'] = 10
params['hidden_dims'] = [ 128, 32 ]
params['output_activation'] = tf.nn.sigmoid
params['activation'] = tf.nn.leaky_relu
params['use_bias'] = True
params['kernel_initializer'] = 'glorot_uniform'
params['bias_initializer'] = 'zeros'
params['kernel_regularizer'] = None
params['bias_regularizer'] = None
params['activity_regularizer'] = None
params['kernel_constraint'] = None
params['bias_constraint'] = None

# loss
params['loss'] = tf.keras.losses.SparseCategoricalCrossentropy()

# training
params['epochs'] = 3
params['mbsize'] = 32
params['optimizer_name'] = 'adam'
params['learning_rate'] = 1e-3

# metrics
params['metrics'] = [tf.keras.metrics.SparseCategoricalAccuracy()]

# Network
ff = FeedForwardNetwork()
ff.build(params)
ff.train(dataset)
