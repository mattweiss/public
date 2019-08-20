from abc import ABC, abstractmethod
import tensorflow as tf
from pdb import set_trace as st

class AbstractDataset(ABC):

    """
    Abstract Base Class for Datasets Library
    """

    def __init__(self, path=None):

        self._path = path
        self._dataset_dict = dict()
                
    ##################
    # Public Methods #
    ##################

    @abstractmethod
    def getDataset(self, vars=None):

        """
        vars - a list of the variables to be returned
        """
    
    ###################
    # Private Methods #
    ###################

    @abstractmethod
    def _loadDataset(self):

        """
        Load Dataset
        """
