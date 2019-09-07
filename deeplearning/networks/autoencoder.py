import numpy as np
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.deeplearning.networks.base import FeedForwardNetwork
from dovebirdia.filtering.kalman_filter import KalmanFilter
from dovebirdia.utilities.base import dictToAttributes

class Autoencoder(FeedForwardNetwork):

    """
    Autoencoder Class
    """
    
    def __init__(self, params=None):

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################
    
    def predict(self, dataset=None):

        pass
        
    def evaluate(self, dataset=None):

        pass

    ###################
    # Private Methods #
    ###################
    
    def _buildNetwork(self):

        # input and output placeholders
        self._X = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='X')
        self._y = tf.placeholder(dtype=tf.float64, shape=(None,self._input_dim), name='y')

        # encoder and decoder
        self._encoder = self._buildDenseLayers(self._X, self._hidden_dims[:-1])
        self._decoder = self._buildDenseLayers(self._encoder, self._hidden_dims[::-1][1:])

        # output layer
        self._X_hat = tf.keras.layers.Dense(units=self._output_dim,
                                            activation=None,
                                            use_bias=self._use_bias,
                                            kernel_initializer=self._kernel_initializer,
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._kernel_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            activity_regularizer=self._activity_regularizer,
                                            kernel_constraint=self._kernel_constraint,
                                            bias_constraint=self._bias_constraint)(self._decoder)
        
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
        self._encoder = self._buildDenseLayers(self._X, self._hidden_dims[:-1])

        # learn z
        self._z = tf.keras.layers.Dense(units=self._output_dim,
                                        activation=None,
                                        use_bias=self._use_bias,
                                        kernel_initializer=self._kernel_initializer,
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._kernel_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activity_regularizer=self._activity_regularizer,
                                        kernel_constraint=self._kernel_constraint,
                                        bias_constraint=self._bias_constraint,
                                        name='z')(self._encoder)

        # learn L, which is vector from which SPD matrix R is formed 
        self._L_dims = np.sum( np.arange( 1, self._hidden_dims[-1] + 1 ) )
        self._L = tf.keras.layers.Dense(units=self._L_dims,
                                        activation=None,
                                        use_bias=self._use_bias,
                                        kernel_initializer=self._kernel_initializer,
                                        bias_initializer=self._bias_initializer,
                                        kernel_regularizer=self._kernel_regularizer,
                                        bias_regularizer=self._bias_regularizer,
                                        activity_regularizer=self._activity_regularizer,
                                        kernel_constraint=self._kernel_constraint,
                                        bias_constraint=self._bias_constraint,
                                        name='L')(self._encoder)

        # learned noise covariance
        self._R = tf.map_fn(self._generate_spd_cov_matrix, self._L)
        
        # Kalman Filter a priori measurement estimate
        self._z_hat_pri = self._kalman_filter.filter([self._z,self._R])['z_hat_pri'][:,0:1,0]
                
        # post kf affine transformation
        self._post_kf_affine = tf.keras.layers.Dense(units=self._hidden_dims[-1],
                                                     activation=None,
                                                     use_bias=self._use_bias,
                                                     kernel_initializer=self._kernel_initializer,
                                                     bias_initializer=self._bias_initializer,
                                                     kernel_regularizer=self._kernel_regularizer,
                                                     bias_regularizer=self._bias_regularizer,
                                                     activity_regularizer=self._activity_regularizer,
                                                     kernel_constraint=self._kernel_constraint,
                                                     bias_constraint=self._bias_constraint)(self._z_hat_pri)   
        # decoder
        self._decoder = self._buildDenseLayers(self._post_kf_affine, self._hidden_dims[::-1][1:])
        
        # output layer
        self._X_hat = tf.keras.layers.Dense(units=self._output_dim,
                                            activation=None,
                                            use_bias=self._use_bias,
                                            kernel_initializer=self._kernel_initializer,
                                            bias_initializer=self._bias_initializer,
                                            kernel_regularizer=self._kernel_regularizer,
                                            bias_regularizer=self._bias_regularizer,
                                            activity_regularizer=self._activity_regularizer,
                                            kernel_constraint=self._kernel_constraint,
                                            bias_constraint=self._bias_constraint)(self._decoder)
       
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
        X = tf.matmul( L, L, transpose_b = True ) + eps * tf.eye( tf.shape(L)[0], dtype=tf.float64 )

        return X
