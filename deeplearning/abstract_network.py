from abc import ABC, abstractmethod
import tensorflow as tf
from pdb import set_trace as st

class AbstractNetwork(ABC):

    """
    Abstract base class for network
    """

    def __init__(self, network_type='abstract'):

        self._network_type = network_type

    ##################
    # Public Methods #
    ##################

    def build(self, params = None ):

        """ 
        Build Network
        TODO: Add parameter list
        """

        self._dict_to_attributes(params)

        # Build Model
        self._buildModel()
            
    @abstractmethod
    def train(self, dataset=None):

        pass
    
    ###################
    # Private Methods #
    ###################

    @abstractmethod
    def _buildModel(self):

        pass

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
        

