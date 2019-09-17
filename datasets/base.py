from abc import ABC, abstractmethod
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.utilities.base import dictToAttributes

class AbstractDataset(ABC):

    """
    Abstract Base Class for Datasets Library
    """

    def __init__(self, params=None):

        #self._path = path
        dictToAttributes(self,params)

        self._data = dict()
                
    ##################
    # Public Methods #
    ##################

    @abstractmethod
    def getDataset(self):

        """
        return dataset
        """
