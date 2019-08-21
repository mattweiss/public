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
    
    def fit(self, dataset=None):

        dictToAttributes(self,dataset)

        self._history = self._model.fit(self._x_train, self._x_train,
                                        batch_size=self._mbsize,
                                        epochs=self._epochs,
                                        validation_data=(self._x_val, self._x_val))

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
        
        self._encoder = self._buildEncoderLayers(encoder_input, self._hidden_dims)
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
        
        self._encoder = self._buildEncoderLayers(encoder_input, self._hidden_dims)
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
                                       bias_constraint=self._bias_constraint,
                                       name='output')(output)

        self._model = tf.keras.Model(inputs=encoder_input, outputs=output, name='autoencoder')

    def _buildEncoderLayers(self, input=None, hidden_dims=None):

        assert input is not None
        assert hidden_dims is not None

        # loop over hidden layers
        for dim_index, dim in enumerate(hidden_dims):

            # pass input parameter on first pass
            output = input if dim_index == 0 else output

            # hidden layer
            output = tf.keras.layers.Dense(units=dim,
                                           activation=self._activation,
                                           use_bias=self._use_bias,
                                           kernel_initializer=self._kernel_initializer,
                                           bias_initializer=self._bias_initializer,
                                           kernel_regularizer=self._kernel_regularizer,
                                           bias_regularizer=self._bias_regularizer,
                                           activity_regularizer=self._activity_regularizer,
                                           kernel_constraint=self._kernel_constraint,
                                           bias_constraint=self._bias_constraint)(output)

        # pass 
        output = tf.keras.layers.Lambda(self._kalman_filter.filter)(output)['x_hat_pri'][:,0,:]
        
        return tf.keras.Model(inputs=input, outputs=output, name='encoder')

    def _buildDecoderLayers(self, input=None, hidden_dims=None):

        return self._buildDenseLayers(input, hidden_dims, name='decoder')
