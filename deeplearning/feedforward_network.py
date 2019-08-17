import tensorflow as tf
from pdb import set_trace as st

from deeplearning.abstract_network import AbstractNetwork

class FeedForwardNetwork(AbstractNetwork):

    """
    Feed Forward Network Class
    """

    def __init__(self, network_type='feedforward'):

        super().__init__(network_type=network_type)

    ##################
    # Public Methods #
    ##################

    def train(self, dataset=None):

        self._dict_to_attributes(dataset)

        history = self._model.fit(self._x_train, self._y_train,
                                  batch_size=self._mbsize,
                                  epochs=self._epochs,
                                  validation_data=(self._x_val, self._y_val))
    
    ###################
    # Private Methods #
    ###################
    
    def _buildModel(self):

        ############################
        # Build Network
        ############################
        self._buildNetwork()
        
        ############################
        # Set Optimizer
        ############################
        self._setOptimizer()

        ############################
        # Compile Network
        ############################
        self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)

    def _buildNetwork(self):

        # set input
        input = tf.keras.Input(shape=(self._input_dim,))
        
        # loop over hidden layers
        for dim_index, dim in enumerate(self._hidden_dims):

            # add output dimension to network params
            #self._network_params['units'] = dim

            # input to hidden layer
            hidden_input = input if dim_index == 0 else output

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
                                           bias_constraint=self._bias_constraint)(hidden_input)

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
            
        self._model = tf.keras.Model(inputs=input, outputs=output)
        print(self._model.summary())
    
    def _setOptimizer(self):

        if self._optimizer_name == 'adam':

            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        
