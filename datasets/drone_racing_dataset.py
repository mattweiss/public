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

class droneRacingDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        datafile : name of text file with ground truth
        datadir: path to text file directory
        features : features to save/load
        n_samples : number of samples to include
        n_steps : spacing between samples
        datasetname: name of dataset

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

    def getDataset(self,load_path=None):

        # load previously saved dataset
        if load_path is not None:

            self._data = self.getSavedDataset(load_path)

        # create and save dataset
        else:

            ########################
            # Read Ground Truth Data
            ########################

            df = pd.read_csv(self._datadir+self._datafile+'/groundtruth.txt', sep=' ', skiprows=None)
            df.sort_values(by=['timestamp'])
            print(df.head())

            #############################
            # Read Data to Data Structure
            #############################

            self._data['y_test'] = df[self._features][:self._n_samples][::self._n_steps].values

            ###################
            # Expand Dimensions
            ###################

            self._data['y_test'] = np.expand_dims(self._data['y_test'],axis=0)
            
            ################################
            # Center Data to Begin at Origin
            ################################

            self._data['y_test'][:,:,0] -= self._data['y_test'][:,0,0]
            self._data['y_test'][:,:,1] -= self._data['y_test'][:,0,1]

            ###########
            # Add Noise
            ###########

            noise = np.random.normal(scale=0.0,size=self._data['y_test'].shape)
            self._data['x_test'] = self._data['y_test'] + noise
            
            #####################
            # Create y_test and t
            #####################

            self._data['t'] = np.linspace(0,self._data['x_test'].shape[1],self._data['x_test'].shape[1])
            
            # plt.figure(figsize=(6,6))
            # plt.scatter(
            #     self._data['x'][:,0],
            #     self._data['x'][:,1],
            #     s=5,
            #     marker='x')
            # plt.grid()
            # plt.show()
            # st()
            
            ###########
            # Save Data
            ###########
            
            save_dir = self._datadir + 'split/' + self._dataset_name.replace(' ','_') + '.pkl'
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        ###################                
        return self._data
