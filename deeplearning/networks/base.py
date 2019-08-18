from abc import ABC, abstractmethod
import tensorflow as tf
from pdb import set_trace as st

class AbstractNetwork(ABC):

    """
    Abstract base class for network
    """

    def __init__(self, params=None):

        """ 
        TODO: Add parameter list
        """
        self._model = None
        self._dict_to_attributes(params)
        
    ##################
    # Public Methods #
    ##################

    def build(self ):

        """ 
        Build Network
        """

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
            
    @abstractmethod
    def train(self, dataset=None):

        pass

    def getModelSummary(self):

        print(self._model.summary())

    
    ###################
    # Private Methods #
    ###################

    @abstractmethod
    def _buildNetwork(self):

        pass
    
    @abstractmethod
    def _setOptimizer(self):

        pass

    def _dict_to_attributes(self,att_dict):

        # Assign Attributes
        for key, value in att_dict.items():

            setattr(self, '_' + key, value)
        
class FeedForwardNetwork(AbstractNetwork):

    """
    Feed Forward Network Class
    """

    def __init__(self, params=None):

        super().__init__(params=params)

    ##################
    # Public Methods #
    ##################

    def train(self, dataset=None):

        self._dict_to_attributes(dataset)

        self._history = self._model.fit(self._x_train, self._y_train,
                                        batch_size=self._mbsize,
                                        epochs=self._epochs,
                                        validation_data=(self._x_val, self._y_val))
    
    ###################
    # Private Methods #
    ###################

    def _buildNetwork(self):

        # set input
        input = tf.keras.Input(shape=(self._input_dim,))

        output = self._buildDenseLayers(input, self._hidden_dims)
        
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
    
    def _setOptimizer(self):

        if self._optimizer_name == 'adam':

            self._optimizer = tf.keras.optimizers.Adam(learning_rate=self._learning_rate)
        
    def _buildDenseLayers(self, input=None, hidden_dims=None):

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

        return tf.keras.Model(inputs=input, outputs=output)
