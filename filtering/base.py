from abc import ABC, abstractmethod
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.utilities.base import dictToAttributes

class AbstractFilter(ABC):

    """
    Abstract base class for filter
    """

    def __init__(self, params=None):

        dictToAttributes(self,params)

    ##################
    # Public Methods #
    ##################
    
    @abstractmethod
    def filter(self):

        pass
        
    ###################
    # Private Methods #
    ###################
