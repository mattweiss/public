from abc import ABC, abstractmethod
import tensorflow as tf
from pdb import set_trace as st

class AbstractFilter(ABC):

    """
    Abstract base class for filter
    """

    def __init__(self, params=None):

        self._dictToAttributes(params=params)

    ##################
    # Public Methods #
    ##################
    
    @abstractmethod
    def filter(self):

        pass
        
    ###################
    # Private Methods #
    ###################
    
    @abstractmethod
    def _dictToAttributes(self,params=None):

        pass
