import numpy as np
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.deeplearning.layers.base import Dense
from dovebirdia.filtering.kalman_filter import KalmanFilter
from dovebirdia.utilities.base import dictToAttributes

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

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')

        # encoder and decoder
        self._encoder = Dense(self._hidden_layer_dict).build(self._X, self._hidden_dims[:-1], scope='encoder')
        self._decoder = Dense(self._hidden_layer_dict).build(self._encoder, self._hidden_dims[::-1][1:], scope='decoder')
        
        # output layer
        self._y_hat = Dense(self._affine_layer_dict).build(self._decoder, [self._output_dim], scope='output')
        
class AutoencoderKalmanFilter(Autoencoder):

    """
    Autoencoder-KalmanFilter Class
    """
    
    def __init__(self, params=None, kf_params=None):

        # instantiate Kalman Filter before parent constructor as
        # the parent calls _buildNetwork()
        self._kalman_filter = KalmanFilter(params=kf_params)
                
        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################
 
    ###################
    # Private Methods #
    ###################
    
    def _buildNetwork(self):
        
        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')

        # encoder
        self._encoder = Dense(self._hidden_layer_dict).build(self._X, self._hidden_dims, scope='encoder')

        # learn z
        self._z = Dense(self._affine_layer_dict).build(self._encoder, [self._hidden_dims[-1]], scope='z')


        # learn L, which is vector from which SPD matrix R is formed 
        self._L_dims = np.sum(np.arange(1, self._hidden_dims[-1] + 1))
        self._L = Dense(self._affine_layer_dict).build(self._encoder, [self._L_dims], scope='L')
        
        # learned noise covariance
        self._R = tf.map_fn(self._generate_spd_cov_matrix, self._L)

        # Kalman Filter a priori measurement estimate
        self._kf_results = self._kalman_filter.filter([self._z,self._R])

        self._z_hat_pri = tf.squeeze(self._kf_results['z_hat_pri'],axis=-1)
        self._z_hat_post = tf.squeeze(self._kf_results['z_hat_post'],axis=-1)

        # post kf affine transformation
        self._post_kf_affine = Dense(self._affine_layer_dict).build(self._z_hat_pri, [self._hidden_dims[-1]], scope='post_kf_affine')

        # decoder
        self._decoder = Dense(self.__dict__).build(self._post_kf_affine, self._hidden_dims[::-1][1:], scope='decoder')

        # output layer
        self._y_hat = Dense(self._output_layer_dict).build(self._decoder, [self._output_dim], scope='y_hat')

        
    def _generate_spd_cov_matrix(self, R):

        """ 
        Generates a symmetric covariance matrix for the input covariance
        given input vector with first n_dims elements the diagonal and the
        remaining elements the off-diagonal elements
        """

        ################################
        # SPD Matrix Based on BPKF Paper
        ################################

        eps = 1e-1

        # initial upper triangular matrix
        L = tf.contrib.distributions.fill_triangular( R, upper = False )
        X = tf.matmul(L,L,transpose_b=True) + eps * tf.eye(tf.shape(L)[0],dtype=tf.float64)

        return X
