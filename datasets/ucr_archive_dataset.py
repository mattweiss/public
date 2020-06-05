import os
import glob
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict

import matplotlib.pyplot as plt

from collections import OrderedDict

from pdb import set_trace as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

class ucrArchiveDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        datadir :  Root of UCR Archive
        dataset: name of dataset
        n_samples : Tuple of samples to include
        test_size : Size of test set, in [0,1]
        random_state : seed for stratified shuffle split
        datasetname : Name of dataset
        """

        # if generating new data
        try:

            super().__init__(params)

        # if loading data
        except:

            pass
            
    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # load previously saved dataset
        if hasattr(self,'_saved_dataset'):

            self._data = self.getSavedDataset(self._saved_dataset)
            
        # create and save dataset
        else:

            ###########
            # Read Data
            ###########

            df_train = pd.read_csv(self._datadir+'raw/{dataset}/{dataset}_TRAIN.tsv'.format(dataset=self._dataset),sep='\t',header=None).fillna(method='pad').values
            df_test = pd.read_csv(self._datadir+'raw/{dataset}/{dataset}_TEST.tsv'.format(dataset=self._dataset),sep='\t',header=None).fillna(method='pad').values

            ###############################
            # Stack Data and Extract Labels
            ###############################

            data = np.vstack([df_train[:,self._n_samples[0]+1:self._n_samples[1]],df_test[:,self._n_samples[0]+1:self._n_samples[1]]])
            labels = np.hstack([df_train[:,0],df_test[:,0]])
            
            #############################
            # Training and Testing Splits
            #############################

            sss = StratifiedShuffleSplit(n_splits=1, test_size=self._test_size, random_state=self._random_state)

            for train_index, test_index in sss.split(data,labels):

                self._data['x_train'], self._data['y_train'] = data[train_index], labels[train_index] 
                self._data['x_test'], self._data['y_test']   = data[test_index], labels[test_index]

                #############
                # Standardize
                #############
                
                scaler = StandardScaler()
                self._data['x_train'] = scaler.fit_transform(self._data['x_train'])
                self._data['x_test'] = scaler.transform(self._data['x_test'])

            ##########
            # Create t
            ##########

            self._data['t'] = np.linspace(0,self._data['x_test'].shape[1],self._data['x_test'].shape[1])
            
            ###########
            # Save Data
            ###########

            save_dir = self._datadir + 'split/' + self._dataset_name.replace(' ','_') + '.pkl'
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        ###################                
        return self._data
