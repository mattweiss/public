import tensorflow as tf
from pdb import set_trace as st
from deeplearning.networks.base import FeedForwardNetwork

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

        self._dictToAttributes(dataset)

        self._history = self._model.fit(self._x_train, self._x_train,
                                        batch_size=self._mbsize,
                                        epochs=self._epochs,
                                        validation_data=(self._x_val, self._x_val))

    ###################
    # Private Methods #
    ###################
    
    def _buildNetwork(self):

        # set inputs
        encoder_input = tf.keras.Input(shape=(self._input_dim,))
        decoder_input = tf.keras.Input(shape=(self._hidden_dims[-1],))
        
        self._encoder = self._buildDenseLayers(encoder_input, self._hidden_dims)
        self._decoder = self._buildDenseLayers(decoder_input, self._hidden_dims[::-1][1:])

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
