import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from itertools import repeat

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
        sample_indices - indices to slice samples
        multi_label - boolean indicating multi label classification or not
        """

        super().__init__(params)

        # cast sensors tuple to list
        try:

            self._sensors = list(self._sensors)

        except:

            self._sensors = np.arange(20)

    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # read training and testing pickle files
        self._pickle_files_train = os.listdir(self._dataset_dir + 'training/')

        if self._with_val:

            self._pickle_files_val = os.listdir(self._dataset_dir + 'validation/')

        self._pickle_files_test = os.listdir(self._dataset_dir + 'testing/')

        # by default all files are read.  If not using synthetic remove here.
        if not self._with_synthetic:

            self._pickle_files_train = [ f for f in self._pickle_files_train if 'synthetic' not in f ]

        # read pickle files and generate pandas dataframes
        self._data_train = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/training/' + pf ) for pf in self._pickle_files_train ] )

        if self._with_val:

            self._data_val  = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/validation/' + pf ) for pf in self._pickle_files_val ] )

        self._data_test  = pd.DataFrame( [ pd.read_pickle( self._dataset_dir + '/testing/' + pf ) for pf in self._pickle_files_test ] )

        ###############
        # training set
        ###############
        self._data['x_train'] = np.asarray([ trial[self._samples[0]:self._samples[1],self._sensors] for trial in self._data_train[self._resistance_type].values ])
        self._data['x_train'] = self._data['x_train'].reshape(-1,  self._data['x_train'].shape[1]*self._data['x_train'].shape[2] )

        if self._labels == 'binary_presence':

            self._data['y_train'] = np.stack(self._data_train.binary_presence_label)

        elif self._labels == 'concentration':

            self._data['y_train'] = np.stack(self._data_train.concentration_label)
            
        #################
        # validation set
        #################
        if self._with_val:

            # validation set
            self._data['x_val'] = np.asarray([ trial[self._samples[0]:self._samples[1],self._sensors] for trial in self._data_val[self._resistance_type].values ])
            self._data['x_val'] = self._data['x_val'].reshape(-1,  self._data['x_val'].shape[1]*self._data['x_val'].shape[2] )
            
            if self._labels == 'binary_presence':

                self._data['y_val'] = np.stack(self._data_val.binary_presence_label)

            elif self._labels == 'concentration':

                self._data['y_val'] = np.stack(self._data_val.concentration_label)
    
        #################
        # test set
        #################
        self._data['x_test'] = np.asarray([ trial[self._samples[0]:self._samples[1],self._sensors] for trial in self._data_test[self._resistance_type].values ])
        self._data['x_test'] = self._data['x_test'].reshape(-1,  self._data['x_test'].shape[1]*self._data['x_test'].shape[2] )

        if self._labels == 'binary_presence':

            self._data['y_test'] = np.stack(self._data_test.binary_presence_label)

        elif self._labels == 'concentration':

            self._data['y_test'] = np.stack(self._data_test.concentration_label)

        #############
        # standardize
        #############

        if self._preprocessing is not None:

            self._sklearn_method(self._preprocessing)
            
        #####
        # PCA
        #####

        if self._pca_components is not 0:

            self._sklearn_method(PCA(n_components=self._pca_components))

        return self._data

    def _sklearn_method(self,method=None):

        assert method is not None
        
        # fit and transform on training data
        self._data['x_train'] = method.fit_transform(self._data['x_train'])

        # transform validation and testing data
        self._data['x_val'] = method.transform(self._data['x_val'])
        self._data['x_test'] = method.transform(self._data['x_test']) 
        
