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
    def fit(self):

        pass

    @abstractmethod
    def evaluate(self):

        pass

    @abstractmethod
    def predict(self):

        pass
    
    ###################
    # Private Methods #
    ###################
