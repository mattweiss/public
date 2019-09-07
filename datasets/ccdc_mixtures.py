import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset

class ccdcMixturesDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:
    
        with_val - include validation set
        resistance_type - raw measurements, z scored, Kalman Filtered, etc.
        labels - analyte ratios to include
        sensors - sensors to include
        with_synthetic - include synthetic data
        """
        
        super().__init__(params)
    
    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # load data
        return self._loadDataset()

    ###################
    # Private Methods #
    ###################

    def _loadDataset(self):

        # read training and testing pickle files
        self._pickle_files_train = os.listdir( self._dataset_dir + 'training/' )
        self._pickle_files_val = os.listdir( self._dataset_dir + 'validation/' )
        self._pickle_files_test = os.listdir( self._dataset_dir + 'testing/' )

        # by default all files are read.  If not using synthetic remove here.
        if not self._with_synthetic:
            
            self._pickle_files_train = [ f for f in self._pickle_files_train if 'synthetic' not in f ]
    
        # read pickle files and generate pandas dataframes
        self._data_train = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/training/' + pf ) for pf in self._pickle_files_train ] )
        self._data_val  = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/validation/' + pf ) for pf in self._pickle_files_val ] )
        self._data_test  = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/testing/' + pf ) for pf in self._pickle_files_test ] )

        # training set
        self._dataset_dict['x_train'] = np.asarray(list(self._data_train[self._resistance_type].values))
        self._dataset_dict['x_train'] = self._dataset_dict['x_train'].reshape(-1,  self._dataset_dict['x_train'].shape[1]* self._dataset_dict['x_train'].shape[2] )
        self._dataset_dict['y_train'] = self._data_train.y

        # validation set
        self._dataset_dict['x_val'] = np.asarray(list(self._data_val[self._resistance_type].values))
        self._dataset_dict['x_val'] = self._dataset_dict['x_val'].reshape(-1,  self._dataset_dict['x_val'].shape[1]* self._dataset_dict['x_val'].shape[2] )
        self._dataset_dict['y_val'] = self._data_val.y
        
        # testing set
        self._dataset_dict['x_test'] = np.asarray(list(self._data_test[self._resistance_type].values))
        self._dataset_dict['x_test'] = self._dataset_dict['x_test'].reshape(-1,  self._dataset_dict['x_test'].shape[1]* self._dataset_dict['x_test'].shape[2] )
        self._dataset_dict['y_test'] = self._data_test.y

        return self._dataset_dict
