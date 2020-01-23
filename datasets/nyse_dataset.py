import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict, loadDict

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
        self._min_ts_len = 1762

        # length of each security data split
        self._split_len = self._min_ts_len

    ##################
    # Public Methods #
    ##################

    def getDataset(self):

        # load previously saved dataset
        if hasattr(self,'_saved_dataset'):

            # self._data = np.load(self._saved_dataset,allow_pickle=True).item()
            self._pkl_data = loadDict(self._saved_dataset)
            self._data = self._pkl_data['data']

        # create and save dataset
        else:

            # read dataset
            self._raw_df = pd.read_csv(self._dataset_dir + 'raw/' + self._csv_file)

            # list of symbols
            self._symbols = self._raw_df.symbol.unique()

            # security data
            self._securities = [ self._raw_df[self._raw_df.symbol==symbol] for symbol in self._symbols \
                                 if self._raw_df[self._raw_df.symbol==symbol].shape[0] == self._min_ts_len ]

            # lists for split data and symbols
            security_splits_list = list()
            symbol_splits_list = list()

            # loop over all symbols
            #for symbol in self._symbols:
            for security in self._securities[:self._n_securities]:

                # get all data for given symbol
                # security = self._raw_df[ self._raw_df.symbol == symbol ]

                # if symbol has sufficient data
                # if security.shape[0] == self._min_ts_len:

                # price data
                security_price_data = security[self._price_types].values

                # split security data into arrays of len self._split_len
                security_price_data_splits = [security_price_data[i * self._split_len:(i + 1) * self._split_len]
                                                for i in range( (len(security_price_data) + self._split_len - 1) // self._split_len )]

                # loop over security data splits
                for security_price_data_split in security_price_data_splits:

                    if security_price_data_split.shape[0] == self._split_len:

                        security_splits_list.append(security_price_data_split[:])
                        symbol_splits_list.append(security.symbol.unique())

            # cast to numpy arrays
            security_splits = np.stack(security_splits_list)
            symbol_splits = np.stack(symbol_splits_list)

            # train/test split
            # train_test_sss = ShuffleSplit(n_splits=1, test_size=self._test_size, random_state=None)

            # for train_idx, test_idx in train_test_sss.split(security_splits, symbol_splits):

            # training and testing data and labels
            train_test_split_index = int(security_splits.shape[1] * 0.8)

            self._data['x_train'] = security_splits[:,:train_test_split_index]
            self._data['x_test'] = security_splits[:,train_test_split_index:]
            self._data['y_labels'] = symbol_splits
            # self._data['y_test'] = symbol_splits[train_test_split_index:]

            # validation set
            if self._with_val is not None:

                train_val_split_index = int(self._data['x_train'].shape[1] * 0.8)

                # temp to hold training values
                x_train_tmp = self._data['x_train']
                # y_train_tmp = self._data['y_train']

                self._data['x_train'] = x_train_tmp[:,:train_val_split_index]
                self._data['x_val'] = x_train_tmp[:,train_val_split_index:]
                # self._data['y_train'] = y_train_tmp[:train_val_split_index]
                # self._data['y_val'] = y_train_tmp[train_val_split_index:]

            # min-max scaler
            if self._feature_range is not None:

                x_train_min_max_list = list()
                x_val_min_max_list = list()
                x_test_min_max_list = list()

                for x_train, x_val, x_test in zip(self._data['x_train'],self._data['x_val'],self._data['x_test']):

                    min_max_scaler = MinMaxScaler(feature_range=(0,1))
                    x_train_min_max_list.append(min_max_scaler.fit_transform(x_train))
                    x_val_min_max_list.append(min_max_scaler.transform(x_val))
                    x_test_min_max_list.append(min_max_scaler.transform(x_test))

                self._data['x_train'] = np.asarray(x_train_min_max_list)
                self._data['x_val'] = np.asarray(x_val_min_max_list)
                self._data['x_test'] = np.asarray(x_test_min_max_list)

            #Baseline shift
            if self._baseline_shift:

                self._data['x_train'] -= self._data['x_train'][:,:1,:]
                self._data['x_val'] -= self._data['x_val'][:,:1,:]
                self._data['x_test'] -= self._data['x_test'][:,:1,:]

            # save to disk
            save_dir = self._dataset_dir + 'split/' + self._dataset_name + '.pkl'

            # np.save(save_dir,self._data,allow_pickle=True)
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        st()

        return self._data
