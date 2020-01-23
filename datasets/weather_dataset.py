import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset

class weatherDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        train_samples: Number of training samples to consider
        with_val: use validation set
        support: support of function,
        feature_range: min max scaling range
        split_len: # Each city's temp has over 45000 samples.  These are split into subsamples of length split_len
        features: temperature,pressure,humidity,wind_direction,wind_speed
        cities: cities to include in dataset

        """

        super().__init__(params)

        # set dataset manually for now
        self._dataset_dir = '/home/mlweiss/Documents/wpi/research/data/weather/'

    ##################
    # Public Methods #
    ##################

    def getDataset(self,np_filename=None):

        # load previously saved dataset
        if hasattr(self,'_saved_dataset'):

            self._data = np.load(self._saved_dataset,allow_pickle=True).item()

        # create and save dataset
        else:

            # function fupport
            # self._data['t'] = np.expand_dims(np.linspace(self._support[0], self._support[1], self._split_len), axis=-1)

            # read dataset
            data_list = list()

            for feature in self._features:

                data_list.append(pd.read_csv(self._dataset_dir + '/raw/' + feature + '.csv', index_col=None, header=0))

            self._raw_df = pd.concat(data_list, axis=1, ignore_index=False)
            # self._raw_df = pd.read_csv(self._dataset_dir + 'raw/' + self._csv_file, usecols=self._cities)

            # fill Nan values
            self._raw_df = self._raw_df.fillna(method='ffill')

            # lists for split data and symbols
            weather_list = list()
            city_list = list()

            # loop over all cities
            for city in self._cities:

                # get all data for given symbol
                weather_values = self._raw_df[city].values[:self._n_samples]
                weather_list.append(weather_values)
                city_list.append(city)

                # split security data into arrays of len self._split_len
                # city_weather_data_splits = [city_weather_data[i * self._split_len:(i + 1) * self._split_len]
                #                                 for i in range( (len(city_weather_data) + self._split_len - 1) // self._split_len )]
                #
                # # loop over security data splits
                # for city_weather_data_split in city_weather_data_splits:
                #
                #     if city_weather_data_split.shape[0] == self._split_len:
                #
                #         weather_splits_list.append(city_weather_data_split[:1500])
                #         city_splits_list.append(city)

            # cast to numpy arrays
            weather_splits = np.stack(weather_list)
            city_splits = np.stack(city_list)

            # training and testing data and labels
            train_test_split_index = int(weather_splits.shape[1] * 0.8)

            self._data['x_train'] = weather_splits[:,:train_test_split_index]
            self._data['x_test'] = weather_splits[:,train_test_split_index:]
            self._data['y_labels'] = city_splits

            # validation set
            if self._with_val is not None:

                train_val_split_index = int(self._data['x_train'].shape[1] * 0.8)

                # temp to hold training values
                x_train_tmp = self._data['x_train']

                self._data['x_train'] = x_train_tmp[:,:train_val_split_index]
                self._data['x_val'] = x_train_tmp[:,train_val_split_index:]

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

            # Baseline shift
            self._data['x_train'] -= self._data['x_train'][:,:1]
            self._data['x_val'] -= self._data['x_val'][:,:1]
            self._data['x_test'] -= self._data['x_test'][:,:1]

            # save to disk
            save_dir = self._dataset_dir + 'split/' + self._dataset_name

            if np_filename is not None:

                save_dir += '_' + np_filename

            np.save(save_dir,self._data,allow_pickle=True)

        return self._data
