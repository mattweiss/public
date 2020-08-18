import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tf_float_prec = tf.float64
from scipy import stats
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.layers.base import Dense
from dovebirdia.math.linalg import pos_diag

try:

    from orthnet import Legendre, Chebyshev

except:

    pass

from sklearn.datasets import make_spd_matrix

from dovebirdia.filtering.kalman_filter import KalmanFilter
from dovebirdia.filtering.interacting_multiple_model import InteractingMultipleModel

from dovebirdia.utilities.base import dictToAttributes, saveDict

class Autoencoder(FeedForwardNetwork):

    """
    Autoencoder Class
    """

    def __init__(self, params=None):

        assert isinstance(params,dict)

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    ###################
   # Private Methods #
    ###################

    def _buildNetwork(self):

        # encoder and decoder
        self._encoder = self._buildEncoder(input=self._X)
        self._decoder = self._buildDecoder(input=self._encoder)

        # output layer
        self._y_hat = self._buildOutput(input=self._decoder)

    def _buildEncoder(self, input=None):

        assert input is not None

        return Dense(self._hidden_layer_dict).build(input, self._hidden_dims[:-1], scope='encoder')

    def _buildDecoder(self, input=None):

        assert input is not None

        return Dense(self._hidden_layer_dict).build(input, self._hidden_dims[::-1][1:], scope='decoder')

    def _buildOutput(self, input=None):

        assert input is not None

        return Dense(self._affine_layer_dict).build(input, [self._output_dim], scope='output')

class AutoencoderKalmanFilter(Autoencoder):

    """
    Autoencoder-KalmanFilter Class
    """

    def __init__(self, params=None, kf_params=None):

        # instantiate Kalman Filter before parent constructor as
        # the parent calls _buildNetwork()
        self._kalman_filter = params['kf_type'](params=kf_params)
        
        super().__init__(params=params)

    ##################
   # Public Methods #
    ##################

    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        self._setPlaceholders()

        self._encoder = self._encoderLayer(self._X)

        self._z, self._R = self._preKalmanFilterAffineLayer(self._encoder)
                
        self._kf_results = self._kalman_filter.fit([self._z,self._R])

        self._post_kf_affine = self._postKalmanFilterAffineLayer(tf.squeeze(self._kf_results['z_hat_pri'],axis=-1))

        self._decoder = self._decoderLayer(self._post_kf_affine)
        
        self._y_hat = self._outputLayer(self._decoder)

    def _setLoss(self):

        super()._setLoss()
      
    def _encoderLayer(self, input=None):

        assert input is not None

        return Dense(name='encoder',
                     weight_initializer=self._weight_initializer,
                     weight_regularizer=self._weight_regularizer,
                     weight_regularizer_scale=self._weight_regularizer_scale,
                     bias_initializer=self._bias_initializer,
                     bias_regularizer=self._bias_regularizer,
                     weight_constraint=self._weight_constraint,
                     bias_constraint=self._bias_constraint,
                     activation=self._activation,
                     use_bias=True,
                     input_dropout_rate=self._input_dropout_rate,
                     dropout_rate=self._dropout_rate).build(input, self._hidden_dims[:-1])

    def _preKalmanFilterAffineLayer(self, input=None):

        assert input is not None

        # scale Z, L and R dimensions if including z dot
        self._dim_scale = self._kalman_filter._model_order + 1 if self._kalman_filter._with_z_dot else 1

        z = Dense(name='z',
                  weight_initializer=self._weight_initializer,
                  weight_regularizer=self._weight_regularizer,
                  weight_regularizer_scale=self._weight_regularizer_scale,
                  bias_initializer=self._bias_initializer,
                  bias_regularizer=self._bias_regularizer,
                  weight_constraint=self._weight_constraint,
                  bias_constraint=self._bias_constraint,
                  activation=None,
                  use_bias=True,
                  input_dropout_rate=0.0,
                  dropout_rate=0.0).build(input, [self._dim_scale*self._hidden_dims[-1]])

        z = tf.expand_dims(z, axis=-1)

        self._L_dims = np.sum(np.arange(1,self._dim_scale*self._hidden_dims[-1]+1))

        # backwards compatibility
        # try:

        # learned noise covariance
        if self._R_model == 'learned':

            self._L = Dense(name='L',
                            weight_initializer=self._weight_initializer,
                            weight_regularizer=self._weight_regularizer,
                            weight_regularizer_scale=self._weight_regularizer_scale,
                            bias_initializer=self._bias_initializer,
                            bias_regularizer=self._bias_regularizer,
                            weight_constraint=self._weight_constraint,
                            bias_constraint=self._bias_constraint,
                            activation=self._R_activation,
                            use_bias=True,
                            input_dropout_rate=0.0,
                            dropout_rate=0.0).build(input, [self._L_dims])

            R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

        elif self._R_model == 'identity':

            R = tf.eye(self._hidden_dims[-1], batch_shape=[tf.shape(z)[0]], dtype=tf_float_prec)
            
        return z, R

    def _kalmanFilterLayer(self, input=None):

        z, R = input

        self._kf_results = self._kalman_filter.fit([z,R])

    def _postKalmanFilterAffineLayer(self, input=None):

        assert input is not None

        return Dense(name='post_kf_affine',
                     weight_initializer=self._weight_initializer,
                     weight_regularizer=self._weight_regularizer,
                     weight_regularizer_scale=self._weight_regularizer_scale,
                     bias_initializer=self._bias_initializer,
                     bias_regularizer=self._bias_regularizer,
                     weight_constraint=self._weight_constraint,
                     bias_constraint=self._bias_constraint,
                     activation=None,
                     use_bias=True,
                     input_dropout_rate=0.0,
                     dropout_rate=self._dropout_rate).build(input, [self._hidden_dims[-2]])

    def _decoderLayer(self, input=None):

        assert input is not None

        return Dense(name='decoder',
                     weight_initializer=self._weight_initializer,
                     weight_regularizer=self._weight_regularizer,
                     weight_regularizer_scale=self._weight_regularizer_scale,
                     bias_initializer=self._bias_initializer,
                     bias_regularizer=self._bias_regularizer,
                     weight_constraint=self._weight_constraint,
                     bias_constraint=self._bias_constraint,
                     activation=self._activation,
                     use_bias=True,
                     input_dropout_rate=0.0,
                     dropout_rate=self._dropout_rate).build(input, self._hidden_dims[::-1][2:]+[self._output_dim])

    def _outputLayer(self, input=None):

        assert input is not None

        return Dense(name='y_hat',
                     weight_initializer=self._weight_initializer,
                     weight_regularizer=self._weight_regularizer,
                     weight_regularizer_scale=self._weight_regularizer_scale,
                     bias_initializer=self._bias_initializer,
                     bias_regularizer=self._bias_regularizer,
                     weight_constraint=self._weight_constraint,
                     bias_constraint=self._bias_constraint,
                     activation=self._output_activation,
                     use_bias=True,
                     input_dropout_rate=0.0,
                     dropout_rate=0.0).build(input, [self._output_dim])

    def _generate_spd_cov_matrix(self, R):

        """
        Generates a symmetric covariance matrix for the input covariance
        given input vector with first n_dims elements the diagonal and the
        remaining elements the off-diagonal elements
        """

        ################################
        # SPD Matrix Based on BPKF Paper
        ################################

        # initial upper triangular matrix
        L = tf.contrib.distributions.fill_triangular(R, upper = True)

        # ensure diagonal of L is positive
        #L = pos_diag(L,diag_func=tf.abs)
        L = tf.multiply(L,tf.math.sign(L))
        
        #eps = 0.0
        R = tf.matmul(L,L,transpose_a=True)
        #+ eps * tf.eye(tf.shape(L)[0],dtype=tf_float_prec)

        return R

    def _make_spd_matrix(self, x):

        """ Wrapper for sklearn make_spd_matrix """

        return tf.py_func(make_spd_matrix, [x], tf_float_prec)

    def _covariance(self,x,y):

        """
        Compute the covariance of x and y
        """
        
        z = tf.stack([tf.reshape(x, [-1]),tf.reshape(y, [-1])],axis=0)

        z -= tf.expand_dims(tf.reduce_mean(z, axis=1), 1)

        scale = tf.cast(tf.shape(z)[1] - 1, tf.float64)

        return tf.matmul(z, tf.transpose(z)) / scale

    def _shapiro_wilk(self, x):

        x = tf.reshape(x, [tf.size(x)])
        print('X.shape in Shapiro-Wilk:%s' % (tf.shape(x)))

        # number of samples
        # n_samples = tf.shape(x, out_type=tf_float_prec)[0]
        n_samples = tf.to_float(tf.shape(x)[0])

        # sample range
        sample_range = tf.range(n_samples, name='sample_range')

        # m_i values
        m_i = tf.map_fn(lambda i: tf_ndtri(tf.divide(tf.subtract(tf.add(i,1.0),0.375),
        tf.add(n_samples,0.25))), sample_range)

        # m
        m = tf.cast(tf.reduce_sum(tf.square(m_i)), dtype=tf_float_prec)

        # u
        u = 1.0 / tf.sqrt(tf.cast(n_samples,dtype=tf_float_prec))

        # a[n-1]
        a_n_1 = tf.multiply(-3.582633,tf.pow(u,5)) + \
        tf.multiply(5.682633,tf.pow(u,4)) - \
        tf.multiply(1.752461,tf.pow(u,3)) - \
        tf.multiply(0.293762,tf.pow(u,2)) + \
        tf.multiply(0.042981,u) + \
        tf.multiply(m_i[-2],tf.pow(m,-0.5))

        # a[n]
        a_n = tf.multiply(-2.706056,tf.pow(u,5)) + \
        tf.multiply(4.434685,tf.pow(u,4)) - \
        tf.multiply(2.071190,tf.pow(u,3)) - \
        tf.multiply(0.147981,tf.pow(u,2)) + \
        tf.multiply(0.221156,u) + \
        tf.multiply(m_i[-1],tf.pow(m,-0.5))

        # stack first two values of a
        a = tf.stack([-a_n,-a_n_1])

        # epsilon
        epsilon = (m - 2.0*m_i[-1]**2 - 2.0*m_i[-2]**2) / (1.0 - 2.0*a_n**2 - 2.0*a_n_1**2)

        # compute a values
        a_range = tf.range(start=2.0, limit=n_samples-2.0, name='a_range')
        a_mid = tf.map_fn(lambda i: m_i[tf.to_int32(i)] / tf.sqrt(epsilon), a_range)

        # concat a values 2 through n-2
        a = tf.concat([a,a_mid],0)

        # concat last two a values
        a = tf.concat([a,tf.expand_dims(a_n_1,axis=0)],0)
        a = tf.concat([a,tf.expand_dims(a_n,axis=0)],0)

        # compute W
        # x = tf.Variable(np.sort(x), name='x', dtype=tf_float_prec)
        x, _ = tf.nn.top_k(x, k=tf.size(x), sorted=True)
        x = tf.cast(x, dtype=tf_float_prec)
        print('X.shape: %s' % (x.shape,))
        W_top = tf.square(tf.reduce_sum(tf.multiply(a,x)))
        W_bottom = tf.reduce_sum(tf.square(x - tf.reduce_mean(x)))

        W = tf.divide(W_top, W_bottom)

        # return W, a, epsilon, u, m, m_i
        return W

class AutoencoderInteractingMultipleModel(AutoencoderKalmanFilter):

    """
    Autoencoder-Interacting Multiple Model Class
    """

    def __init__(self, params=None, kf_params=None):

        super().__init__(params=params,kf_params=kf_params)
        
    ##################
    # Public Methods #
    ##################

    ###################
    # Private Methods #
    ###################

    def _setLoss(self):

        super()._setLoss()
