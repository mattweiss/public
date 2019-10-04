import tensorflow as tf
from dovebirdia.utilities.base import dictToAttributes
from pdb import set_trace as st

class DenseLayer():

    def __init__(self,
                 params=None
                 # weight_initializer=None,
                 # weight_regularizer=None,
                 # bias_initializer=None,
                 # bias_regularizer=None,
                 # activation=None,
                 # use_bias=True
    ):

        assert isinstance(params,dict)
        dictToAttributes(self,params)

    def build(self, x=None, dims=None, scope=None, dropout_rate=0.0):

        assert x is not None
        assert isinstance(dims, list)
        assert scope is not None

        for dim_idx, dim in enumerate(dims):

            w_name = 'W{dim_idx}'.format(dim_idx = dim_idx + 1)

            with tf.variable_scope(name_or_scope=scope+'/weight',
                                   regularizer=self._weight_regularizer,
                                   reuse=tf.AUTO_REUSE):

                W = tf.get_variable(name=w_name,
                                    shape=(x.get_shape()[1],dim),
                                    initializer=self._weight_initializer,
                                    trainable=True,
                                    dtype=tf.float64)

            if self._use_bias:

                b_name = 'b{dim_idx}'.format(dim_idx = dim_idx + 1)

                with tf.variable_scope(name_or_scope=scope+'/bias',
                                       regularizer=self._bias_regularizer,
                                       reuse=tf.AUTO_REUSE):

                    b = tf.get_variable(name=b_name,
                                        shape=dim,
                                        initializer=self._bias_initializer,
                                        trainable=True,
                                        dtype=tf.float64)

            x = tf.matmul(x,W)

            if self._use_bias:

                x = tf.add(x,b)
                
            if self._activation is not None:

                x = self._activation(x)

            # if dropout_rate is not 0.0:

            #     x = tf.nn.dropout(x, rate=dropout_rate)
            
        return x
