from abc import ABC, abstractmethod
import tensorflow as tf
from pdb import set_trace as st
from dovebirdia.utilities.base import dictToAttributes
from dovebirdia.utilities.base import loadDict

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
    def getDataset(self,load_path=None):

        """
        return dataset
        """
        pass

    # Common method to return previously created and saved dataset
    def getSavedDataset(self,dataset_name=None):

        assert dataset_name is not None

        self._pkl_data = loadDict(dataset_name)
        return self._pkl_data['data']
