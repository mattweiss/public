import numpy as np              
import tensorflow as tf
from scipy import stats

from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.layers.base import Dense

from orthnet import Legendre

from pdb import set_trace as st

class HilbertFilter(FeedForwardNetwork):

    """
    Implementation of network that learns coefficients of functions in Hilbert Space
    """

    def __init__(self, params=None):

        assert params is not None

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        # placeholders
        self._setPlaceholders()

        # hidden layers
        self._alpha = Dense(name='hidden_layers',
                             weight_initializer=self._weight_initializer,
                             weight_regularizer=self._weight_regularizer,
                             weight_regularizer_scale=self._weight_regularizer_scale,
                             bias_initializer=self._bias_initializer,
                             bias_regularizer=self._bias_regularizer,
                             weight_constraint=self._weight_constraint,
                             bias_constraint=self._bias_constraint,
                             activation=self._activation,
                             use_bias=self._use_bias,
                             input_dropout_rate=self._input_dropout_rate,
                             dropout_rate=self._dropout_rate).build(self._X,
                                                                    self._hidden_dims+[self._coeff_dim])

        # Fourier coefficients
        #self._alpha = tf.expand_dims(tf.reduce_mean(self._hidden,axis=0),axis=1)
        #self._alpha = tf.transpose(tf.norm(self._hidden,ord=2,axis=0,keepdims=True))
        #self._alpha = tf.matmul(tf.transpose(self._hidden),self._hidden)
        
        # self._alpha = Dense(name='output_layer',
        #                      weight_initializer=self._weight_initializer,
        #                      weight_regularizer=self._weight_regularizer,
        #                      weight_regularizer_scale=self._weight_regularizer_scale,
        #                      bias_initializer=self._bias_initializer,
        #                      bias_regularizer=self._bias_regularizer,
        #                      weight_constraint=self._weight_constraint,
        #                      bias_constraint=self._bias_constraint,
        #                      activation=self._output_activation,
        #                      use_bias=True,
        #                      input_dropout_rate=self._input_dropout_rate,
        #                      dropout_rate=self._dropout_rate).build(self._alpha,
        #                                                             [1])


        # orthonormal basis
        self._polyBase = tf.transpose(Legendre(self._t,self._coeff_dim-1).tensor)

        # function approximation
        self._y_hat = tf.matmul(self._alpha,self._polyBase)

        #self._y_hat = tf.transpose(tf.matmul(self._polyBase,self._alpha,transpose_b=True))
        
        #self._y_hat = self._hidden
        
        #self._hidden = tf.reshape(self._hidden,(-1,self._coeff_dim,self._output_dim))
        # self._alpha = tf.expand_dims(tf.reduce_mean(self._hidden,axis=0),axis=-1)

        # #self._polyBase = tf.map_fn(lambda X : Legendre(tf.expand_dims(tf.transpose(X),axis=-1),self._coeff_dim-1).tensor, elems=tf.transpose(self._X))
        # #self._polyBase = tf.reshape(self._polyBase, (-1,self._coeff_dim,self._output_dim))
        # self._polyBase = Legendre(self._t,self._coeff_dim-1).tensor

        # # function approximation
        # #self._y_hat = tf.transpose(tf.map_fn(lambda alpha : tf.matmul(self._polyBase,alpha))), [self._alpha], dtype=tf.float64))
        # self._y_hat = tf.matmul(self._polyBase,self._alpha)
        # #self._y_hat = tf.reshape(self._y_hat,(-1,self._output_dim))

    def _setPlaceholders(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')
        self._t = tf.placeholder(dtype=tf.float64, shape=(self._input_dim,1), name='t')
        self._mask = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='mask')
