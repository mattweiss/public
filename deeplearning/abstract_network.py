from abc import ABC, abstractmethod

class AbstractNetwork(ABC):

    """
    Abstract base class for network
    """

    def __init__(self,
                 model='sequential'):

        self_model = model

    ##################
    # Public Methods #
    ##################

    def compile(self,
                params=None,
                loss=None,
                optimizer=None):

        """ 
        params : dictionary where keys are input_dim, output_dim, hidden_dims and (3) all keys matching arguments of tf.keras.Dense apart from units which is contained in hidden_dims
        loss : loss function
        optimier : optimizer function
        
        """
        
        assert params is not None
        assert loss is not None
        assert optimizer is not None

        self._input_dim = params.pop('input_dim')
        self._output_dim = params.pop('output_dim')
        self._network_params = params
        
        self._loss = loss
        self._optimizer = optimizer
        
        self._buildNetwork()
        self._setLoss()
        self._setOptimizer()

    @abstractmethod
    def train(self, dataset=None):

        pass
    
    ###################
    # Private Methods #
    ###################
    
    @abstractmethod
    def _buildNetwork(self):

        pass

    @abstractmethod
    def _setLoss(self):

        pass

    @abstractmethod
    def _setOptimizer(self):

        pass

