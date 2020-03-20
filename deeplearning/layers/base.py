import tensorflow as tf
from dovebirdia.utilities.base import dictToAttributes
from pdb import set_trace as st

class Dense():

    def __init__(self,
                 name=None,
                 weight_initializer=None,
                 weight_regularizer=None,
                 weight_regularizer_scale=0.0,
                 bias_initializer=None,
                 bias_regularizer=None,
                 weight_constraint=None,
                 bias_constraint=None,
                 activation=None,
                 use_bias=True,
                 dropout_rate=0.0,
                 input_dropout_rate=0.0
    ):

        #assert isinstance(params,dict)
        #dictToAttributes(self,params)

        assert name is not None

        self._name=name
        self._weight_initializer=weight_initializer
        self._bias_initializer=bias_initializer
        self._bias_regularizer=bias_regularizer
        self._weight_constraint=weight_constraint
        self._bias_constraint=bias_constraint
        self._activation=activation
        self._use_bias=use_bias
        self._dropout_rate=dropout_rate
        self._input_dropout_rate=input_dropout_rate

        # weight regularizer
        try:

            self._weight_regularizer = weight_regularizer(weight_regularizer_scale)

        except:

            self._weight_regularizer = None

    def build(self, x=None, dims=None):

        assert x is not None
        assert isinstance(dims, list)

        if self._input_dropout_rate is not 0.0:

            x = tf.nn.dropout(x, rate=self._input_dropout_rate)

        for dim_idx, dim in enumerate(dims):

            w_name = 'W{dim_idx}'.format(dim_idx = dim_idx + 1)

            with tf.variable_scope(name_or_scope=self._name+'/weight',
                                   regularizer=self._weight_regularizer,
                                   reuse=tf.AUTO_REUSE):

                W = tf.get_variable(name=w_name,
                                    shape=(x.get_shape()[1],dim),
                                    initializer=self._weight_initializer,
                                    constraint=self._weight_constraint,
                                    trainable=True,
                                    dtype=tf.float64)

            if self._use_bias:

                b_name = 'b{dim_idx}'.format(dim_idx = dim_idx + 1)

                with tf.variable_scope(name_or_scope=self._name+'/bias',
                                       regularizer=self._bias_regularizer,
                                       reuse=tf.AUTO_REUSE):

                    b = tf.get_variable(name=b_name,
                                        shape=dim,
                                        initializer=self._bias_initializer,
                                        constraint=self._bias_constraint,
                                        trainable=True,
                                        dtype=tf.float64)

            x = tf.matmul(x,W)

            if self._use_bias:

                x = tf.add(x,b)

            if self._activation is not None:

                x = self._activation(x)

            if self._dropout_rate is not 0.0:

                x = tf.nn.dropout(x, rate=self._dropout_rate)

        return x
