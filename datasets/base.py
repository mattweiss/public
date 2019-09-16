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

        self._dataset_dict = dict()
                
    ##################
    # Public Methods #
    ##################

    @abstractmethod
    def getDataset(self):

        """
        return dataset
        """
    
    ###################
    # Private Methods #
    ###################

    @abstractmethod
    def _loadDataset(self):

        """
        Load Dataset
        """
    
    def _saveDataset(self):

        """
        Save Dataset
        
        Not an abstract method since some datasets, e.g. mnist, are not saved.
        
        """

        pass
