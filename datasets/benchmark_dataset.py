import os
import glob
import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict, generateMask

import matplotlib.pyplot as plt

from collections import OrderedDict

class benchmarkDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        datadir: path to text file directory
        datafile: name of text file with sonar data
        datafile_true : name of text file with ground truth
        r : tuple of inner and outer radii to use for generating band
        meas_dims : Maximum number of measurements per observation
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

            ###################################
            # Read raw data
            ###################################
            data = np.genfromtxt(self._datadir+'raw/'+self._datafile, delimiter=' ')
            data_true = np.genfromtxt(self._datadir+'raw/'+self._datafile_true, delimiter=' ')

            ###################################
            # Slice true data for each target
            ###################################
            # true time
            t = data_true[1::6,0].astype(int)

            #bearing_data_true_stack = 0.0

            # targets
            targets = [
                        'Target 1',
                        'Target 2',
                        'Target 3',
                        'Target 4'
            ]

            # stack target data
            for target_index,target in enumerate(targets):

                try:

                    data_true_stack = np.vstack((data_true_stack,
                                                 data_true[target_index+1::6,1]))

                except:

                    data_true_stack = data_true[target_index+1::6,1]

            data_true_stack = data_true_stack.T

            if np.ndim(data_true_stack) == 1:

                data_true_stack = np.expand_dims(data_true_stack,axis=-1)

            ###################################
            # Band data
            ###################################
            # upper and lower bounds of sonar band
            r0, r1 = self._r[0], self._r[1]

            data_band = list()

            # loop over sonar and truth
            for t_index, t_val in enumerate(t):

                # get all sonar data for each time
                data_t = data[data[:,0]==t_val][:,1]

                for target_index,target in enumerate(targets):

                    # select ground truth at each time
                    data_true_t = data_true_stack[t_index,target_index]

                    band = data_t[(np.abs(data_t-data_true_t) > r0) & 
                                  (np.abs(data_t-data_true_t) < r1) ]

                    if band.shape[0] > self._meas_dims:
            
                        data_band.append(np.random.choice(band,self._meas_dims,replace=False))
    
                    else:

                        # end pad with zeros up to self._meas_dims
                        data_band.append(np.pad(band,(0,self._meas_dims-band.shape[0]),'constant'))
    
            data_band = np.reshape(data_band,(-1,len(targets)))

            ################################################
            # Write to data dictionary and add noise to data
            ################################################
            self._data['y_test'] = data_true_stack
            self._data['x_test'] = data_band
            #self._data['keys'] = [ k for k,v in data_dict.items() if v.shape[0] > self._max_len ]
            self._data['t'] = t

            ##############
            # save to disk
            ##############

            save_dir = self._datadir + 'split/' + self._dataset_name.replace(' ','_') + '.pkl'
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        ###################                
        return self._data
