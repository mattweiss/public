import os
import numpy as np
import random
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict

class nyseDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        train_size - size of training set
        with_val - boolean
        price_types - open, close, etc.
        symbol(s) - list of symbols to consider
        """

        super().__init__(params)

        # set dataset manually for now
        self._dataset_dir = '/home/mlweiss/Documents/wpi/research/data/nyse/'
        self._csv_file = 'prices-split-adjusted.csv'

        # minimum length of time series for inclusion in dataset
        self._min_ts_len = 500

        # length of each security data split
        self._split_len = self._min_ts_len

    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # load previously saved dataset
        if hasattr(self,'_saved_dataset'):

            self._data = self.getSavedDataset(self._saved_dataset)

        # create and save dataset
        else:

            # read dataset
            self._raw_df = pd.read_csv(self._dataset_dir + 'raw/' + self._csv_file)

            # security data
            self._securities = [ self._raw_df[self._raw_df.symbol==symbol][:self._n_samples] for symbol in self._raw_df.symbol.unique() \
                                 if self._raw_df[self._raw_df.symbol==symbol].shape[0] >= self._min_ts_len ]

            # lists for split data and symbols
            security_list = list()
            symbol_list = list()

            # loop over all symbols
            if self._n_securities is not None:

                for security in random.sample(population=self._securities,k=self._n_securities):

                    if security[self._price_types].values.shape[0] == 1762:
                    
                        security_list.append(security[self._price_types].values)
                        symbol_list.append(security.symbol.unique())

            else:

                for security in self._securities:

                    if security[self._price_types].values.shape[0] == 1762:
                    
                        security_list.append(security[self._price_types].values)
                        symbol_list.append(security.symbol.unique())
                
            securities = np.stack(security_list)
            symbols = np.stack(symbol_list)

            ######################
            # Train-Val-Test Split
            ######################

            # training and testing data and labels
            train_test_split_index = int(securities.shape[1] * 0.8)

            self._data['x_train'] = securities[:,:train_test_split_index]
            self._data['x_test'] = securities[:,train_test_split_index:]
            self._data['y_labels'] = symbols

            # validation set
            if self._with_val is not None:

                train_val_split_index = int(self._data['x_train'].shape[1] * 0.8)

                # temp to hold training values
                x_train_tmp = self._data['x_train']

                self._data['x_train'] = x_train_tmp[:,:train_val_split_index]
                self._data['x_val'] = x_train_tmp[:,train_val_split_index:]

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

            ###############
            # MinMax Scaler
            ###############

            if self._feature_range is not None:
            
                min_max_scaler = MinMaxScaler(feature_range=self._feature_range)
                self._data['x_train'] = min_max_scaler.fit_transform(self._data['x_train'])
                self._data['x_val'] = min_max_scaler.transform(self._data['x_val'])
                self._data['x_test'] = min_max_scaler.transform(self._data['x_test'])

            ##################
            # Reshape Datasets
            ##################

            self._data['x_train'] = self._data['x_train'].reshape(train_shape_orig)
            self._data['x_val'] = self._data['x_val'].reshape(val_shape_orig)
            self._data['x_test'] = self._data['x_test'].reshape(test_shape_orig)

            for trial in np.arange(self._data['x_train'].shape[0]):

                self._data['x_train'][trial] -= self._data['x_train'][trial,0,:]
                self._data['x_val'][trial] -= self._data['x_val'][trial,0,:]
                self._data['x_test'][trial] -= self._data['x_test'][trial,0,:]
                
            # save to disk
            save_dir = self._dataset_dir + 'split/' + self._dataset_name + '.pkl'

            # np.save(save_dir,self._data,allow_pickle=True)
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        return self._data
