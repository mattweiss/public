import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict

class s5Dataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        n_samples - Number of samples per time series
        benchmark - which benchmark dataset to use
        dataset_name - name of output file

        """

        super().__init__(params)

        
    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # load previously saved dataset
        if hasattr(self,'_saved_dataset'):

            self._data = self.getSavedDataset(self._saved_dataset)

        # create and save dataset
        else:

            # set dataset manually for now
            self._dataset_dir = '/home/mlweiss/Documents/wpi/research/data/anomaly/s5/ydata-labeled-time-series-anomalies-v1_0/'
            self._csv_full_path = self._dataset_dir + 'raw/' + self._benchmark + 'Benchmark'

            # set maximum samples if using A1 dataset, otherwise set to None
            self._max_samples = 1400 if self._benchmark == 'A1' else None
        
            ###########
            # Load Data
            ###########

            # if file names contain underscore
            try:

                csv_files = sorted(glob.glob(os.path.join(self._csv_full_path, '*.csv')),
                                    key = lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]))

            # if files names contain hyphen
            except:

                csv_files = sorted(glob.glob(os.path.join(self._csv_full_path, '*.csv')),
                                    key = lambda x: x.split('/')[-1].split('.')[0].split('-')[1])

            # list of pandas data frames for each time series in dataset
            data = [ pd.read_csv(csv_file) for csv_file in csv_files ]
            
            # rename is_anomaly columns to anomaly
            try:

                data = [ df.rename(columns={'is_anomaly':'anomaly'}) for df in data ]

            except:

                pass

            #########################################
            # create x, y and t entries in self._data
            #########################################

            x_list = list()
            y_list = list()
            
            for d in data:

                # if A1 benchmark
                try:
                    
                    if d.shape[0] >= self._max_samples:
                
                        x_list.append(np.expand_dims(d['value'].values[:self._max_samples],axis=-1))
                        y_list.append(d['anomaly'][:self._max_samples].values)

                except:

                    x_list.append(np.expand_dims(d['value'].values[:self._max_samples],axis=-1))
                    y_list.append(d['anomaly'].values)

            x = np.asarray(x_list)
            y = np.asarray(y_list)

            ######################
            # Train-Val-Test Split
            ######################

            #training and testing data and labels
            train_test_split_index = 500 #int(y.shape[0] * 0.8)

            self._data['x_train'] = x[:,:train_test_split_index]
            self._data['y_train'] = y[:,:train_test_split_index]
            
            self._data['x_test'] = x[:,train_test_split_index:]
            self._data['y_test'] = y[:,train_test_split_index:]
            
            # validation set
            train_val_split_index = 400 #int(self._data['y_train'].shape[0] * 0.8)

            x_train_tmp = self._data['x_train']
            y_train_tmp = self._data['y_train']

            self._data['x_train'] = x_train_tmp[:,:train_val_split_index]
            self._data['x_val'] = x_train_tmp[:,train_val_split_index:]
            
            self._data['y_train'] = y_train_tmp[:,:train_val_split_index]
            self._data['y_val'] = y_train_tmp[:,train_val_split_index:]

            ###########################
            # Reshape for preprocessing
            ###########################

            train_shape_orig, val_shape_orig, test_shape_orig = self._data['x_train'].shape, self._data['x_val'].shape, self._data['x_test'].shape

            self._data['x_train'] = self._data['x_train'].reshape(self._data['x_train'].shape[0]*self._data['x_train'].shape[1],self._data['x_train'].shape[2])
            self._data['x_val'] = self._data['x_val'].reshape(self._data['x_val'].shape[0]*self._data['x_val'].shape[1],self._data['x_val'].shape[2])
            self._data['x_test'] = self._data['x_test'].reshape(self._data['x_test'].shape[0]*self._data['x_test'].shape[1],self._data['x_test'].shape[2])
            
            ##################
            # Standardize Data
            ##################

            if self._standardize:
            
                standard_scaler = StandardScaler()
                self._data['x_train'] = standard_scaler.fit_transform(self._data['x_train'])
                self._data['x_val'] = standard_scaler.transform(self._data['x_val'])
                self._data['x_test'] = standard_scaler.transform(self._data['x_test'])

            ##################
            # Reshape Datasets
            ##################

            self._data['x_train'] = self._data['x_train'].reshape(train_shape_orig)
            self._data['x_val'] = self._data['x_val'].reshape(val_shape_orig)
            self._data['x_test'] = self._data['x_test'].reshape(test_shape_orig)
            
            # save to disk
            save_dir = self._dataset_dir + 'split/' + self._dataset_name + '.pkl'

            # np.save(save_dir,self._data,allow_pickle=True)
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        return self._data
