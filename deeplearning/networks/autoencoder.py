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
    
    def fit(self, dataset=None, save_weights=False):

        dictToAttributes(self,dataset)

        self._history = self._model.fit(self._x_train, self._x_train,
                                        batch_size=self._mbsize,
                                        epochs=self._epochs,
                                        validation_data=(self._x_val, self._x_val))

        if save_weights:

            self._model.save_weights(self._results_dir + self._model_name + '.keras')

    def predict(self, dataset=None):

        pass
        
    def evaluate(self, dataset=None):

        pass

    ###################
    # Private Methods #
    ###################
    
    def _buildNetwork(self):

        # set inputs
        encoder_input = tf.keras.Input(shape=(self._input_dim,))
        decoder_input = tf.keras.Input(shape=(self._hidden_dims[-1],))
        
        self._encoder = self._buildEncoderLayers(encoder_input, self._hidden_dims[:-1])
        self._decoder = self._buildDecoderLayers(decoder_input, self._hidden_dims[::-1][1:])

        # decoder output
        output = self._decoder(self._encoder(encoder_input))
        
        # output layer
        output = tf.keras.layers.Dense(units=self._output_dim,
                                       activation=self._output_activation,
                                       use_bias=self._use_bias,
                                       kernel_initializer=self._kernel_initializer,
                                       bias_initializer=self._bias_initializer,
                                       kernel_regularizer=self._kernel_regularizer,
                                       bias_regularizer=self._bias_regularizer,
                                       activity_regularizer=self._activity_regularizer,
                                       kernel_constraint=self._kernel_constraint,
                                       bias_constraint=self._bias_constraint)(output)
            
        self._model = tf.keras.Model(inputs=encoder_input, outputs=output)

    def _buildEncoderLayers(self, input=None, hidden_dims=None):

        return self._buildDenseLayers(input, hidden_dims, name='encoder')

    def _buildDecoderLayers(self, input=None, hidden_dims=None):

        return self._buildDenseLayers(input, hidden_dims, name='decoder')

    
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

        # set inputs
        encoder_input = tf.keras.Input(shape=(self._input_dim,), dtype=tf.float64, name='encoder_input')
        decoder_input = tf.keras.Input(shape=(self._hidden_dims[-1],), dtype=tf.float64, name='decoder_input')

        # encoder/decoder models
        self._encoder = self._buildEncoderLayers(encoder_input, self._hidden_dims[:-1])
        self._decoder = self._buildDecoderLayers(decoder_input, self._hidden_dims[::-1][1:] + [self._input_dim])

        # Map z
        z = tf.keras.layers.Dense(units=self._hidden_dims[-1],
                                  activation=None,
                                  use_bias=self._use_bias,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer,
                                  kernel_regularizer=self._kernel_regularizer,
                                  bias_regularizer=self._bias_regularizer,
                                  activity_regularizer=self._activity_regularizer,
                                  kernel_constraint=self._kernel_constraint,
                                  bias_constraint=self._bias_constraint,
                                  name='z')(self._encoder(encoder_input))

        # Map L to R
        self._L_dims = np.sum( np.arange( 1, self._hidden_dims[-1] + 1 ) )
        
        L = tf.keras.layers.Dense(units=self._L_dims,
                                  activation=None,
                                  use_bias=self._use_bias,
                                  kernel_initializer=self._kernel_initializer,
                                  bias_initializer=self._bias_initializer,
                                  kernel_regularizer=self._kernel_regularizer,
                                  bias_regularizer=self._bias_regularizer,
                                  activity_regularizer=self._activity_regularizer,
                                  kernel_constraint=self._kernel_constraint,
                                  bias_constraint=self._bias_constraint,
                                  name='L')(self._encoder(encoder_input))

        R = tf.keras.layers.Lambda(self._generate_input_cov, name='R')(L)

        kf_output = tf.keras.layers.Lambda(self._kalman_filter.filter, name='KF')([z,R])['z_hat_pri'][:,0:1,0]

        # Post KF Affine Transformation
        output = tf.keras.layers.Dense(units=self._hidden_dims[-1],
                                       activation=None,
                                       use_bias=self._use_bias,
                                       kernel_initializer=self._kernel_initializer,
                                       bias_initializer=self._bias_initializer,
                                       kernel_regularizer=self._kernel_regularizer,
                                       bias_regularizer=self._bias_regularizer,
                                       activity_regularizer=self._activity_regularizer,
                                       kernel_constraint=self._kernel_constraint,
                                       bias_constraint=self._bias_constraint,
                                       name='post_kf_affine')(kf_output)
        
        # decoder
        output = self._decoder(output)
        
        # output layer
        output = tf.keras.layers.Dense(units=self._output_dim,
                                       activation=self._output_activation,
                                       use_bias=self._use_bias,
                                       kernel_initializer=self._kernel_initializer,
                                       bias_initializer=self._bias_initializer,
                                       kernel_regularizer=self._kernel_regularizer,
                                       bias_regularizer=self._bias_regularizer,
                                       activity_regularizer=self._activity_regularizer,
                                       kernel_constraint=self._kernel_constraint,
                                       bias_constraint=self._bias_constraint,
                                       name='output')(output)

        self._model = tf.keras.Model(inputs=encoder_input, outputs=output, name='autoencoder')

    # These two functions are nested because tf.keras.layers.Lambda does not work if tf.map_fn is passed to it.
    def _generate_input_cov(self, L):

        return tf.map_fn(self._generate_spd_cov_matrix, L)
    
    def _generate_spd_cov_matrix(self, R):

        """ Generates a symmetric covariance matrix for the input covariance
            given input vector with first n_dims elements the diagonal and the
            remaining elements the off-diagonal elements """

        ################################
        # SPD Matrix Based on BPKF Paper
        ################################

        eps = 1e-1

        # initial upper triangular matrix
        L = tf.contrib.distributions.fill_triangular( R, upper = False )
        X = tf.matmul( L, L, transpose_b = True ) + eps * tf.eye( tf.shape(L)[0], dtype=tf.float64 )

        return X
