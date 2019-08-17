import tensorflow as tf

from deeplearning.abstract_network import AbstractNetwork
from deeplearning.feedforward_network import FeedForwardNetwork

ff = FeedForwardNetwork()
print(ff.__class__)

# parameters dictionary
params = dict()
params['input_dim'] = 784
params['output_dim'] = 10
params['hidden_dims'] = [ 128, 64 ]
params['activation'] = tf.nn.leaky_relu
params['use_bias'] = True
params['kernel_initializer'] = 'glorot_uniform'
params['bias_initializer'] = 'zeros'
params['kernel_regularizer'] = None
params['bias_regularizer'] = None
params['activity_regularizer'] = None
params['kernel_constraint'] = None
params['bias_contraint'] = None

# loss function
loss = tf.keras.losses.MeanSquaredError

# optimizer
optimizer = tf.keras.optimizers.Adam

ff.compile( params, loss, optimizer )
