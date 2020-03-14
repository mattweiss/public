import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pdb import set_trace as st
from dovebirdia.datasets.base import AbstractDataset
from dovebirdia.utilities.base import saveAttrDict

class weatherDataset(AbstractDataset):

    def __init__(self, params=None):

        """
        Parameters:

        datadir: path to data set csv files
        samples: tuple of samples to include (start,end)
        city: city to include in dataset
        features: temperature,pressure,humidity,wind_direction,wind_speed
        standardize: whether to use StandardScaler, bool

        """

        super().__init__(params)

    ##################
    # Public Methods #
    ##################

    def getDataset(self,np_filename=None):

        # load previously saved dataset
        if hasattr(self,'_saved_dataset'):

            self._data = self.getSavedDataset(self._saved_dataset)

        # create and save dataset
        else:

            ###################################
            # Read CSV Data to Pandas Dataframe
            ###################################
            
            # find all csv files in data directory
            csv_paths = glob.glob(os.path.join(self._datadir+'raw/', '*.csv'))

            # empty pandas dataframe
            data_pd = pd.DataFrame()

            for csv_path in csv_paths:
    
                feature = csv_path.split('/')[-1].split('.')[0]
    
                if feature in self._features:

                    # if first time concatenating, include datetime column
                    if data_pd.shape[0] == 0:
            
                        read_columns = ['datetime',self._city]
            
                    else:
            
                        read_columns = self._city
        
                    # load weather features for given city, ignoring first row as it often contains NAN values
                    data_pd = pd.concat([data_pd,pd.read_csv(csv_path,skiprows=[1])[read_columns]],axis=1)

                    # rename columns
                    column_names = {
                        self._city:feature
                    }

                    # Rename Columns to match features
                    data_pd = data_pd.rename(columns=column_names)

            # convert datetime column to datetime data type
            data_pd['datetime'] = pd.to_datetime(data_pd['datetime'])

            # set datetime column as index and remove datetime column
            data_pd = data_pd.set_index(data_pd['datetime'])
            data_pd = data_pd.drop(columns='datetime')

            # interpolate NANn values
            data_pd = data_pd.interpolate(method ='time', limit_direction ='forward', limit = 10)

            #################################
            # Pandas Dataframe to Numpy Array
            #################################

            X = data_pd.to_numpy()[self._n_samples[0]:self._n_samples[1]]
            self._features = data_pd.columns
            
            ######################
            # Train-Val-Test Split
            ######################

            #train-test
            train_test_split_indices = int(0.8*X.shape[0])
            X_train, X_test = X[:train_test_split_indices], X[train_test_split_indices:]

            # train-val
            train_val_split_indices = int(0.8*X_train.shape[0])
            X_train, X_val = X_train[:train_val_split_indices], X_train[train_val_split_indices:]

            assert np.sum(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])==X.shape[0]
            
            ##################
            # Standardize Data
            ##################

            if self._standardize:
                
                scaler = StandardScaler(with_mean=True,with_std=True)
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

                assert np.sum(X_train.shape[0]+X_val.shape[0]+X_test.shape[0])==X.shape[0]

            ######################
            # Assign to attributes
            ######################

            self._data['x_train'] = np.expand_dims(X_train,axis=0)
            self._data['x_val'] = np.expand_dims(X_val,axis=0)
            self._data['x_test'] = np.expand_dims(X_test,axis=0)

            ##############
            # save to disk
            ##############

            save_dir = self._datadir + 'split/' + self._dataset_name.replace(' ','_') + '.pkl'

            # np.save(save_dir,self._data,allow_pickle=True)
            saveAttrDict(save_dict=self.__dict__, save_path=save_dir)

        return self._data
