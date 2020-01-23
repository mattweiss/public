import os, sys
from abc import ABC, abstractmethod
import pandas as pd
from pdb import set_trace as st

class SyntheticSensorDataGenerator(ABC):

    """ Abstract base class for synthetic sensor data """

    def __init__( self,
                  dataset_dir = None,
                  trials = None,
                  sensors = None,
                  labels = None,
                  n_synthetic_sensors_per_label = 1,
                  n_max_samples = 4900,
                  baseline_length = 500,
                  save_plots = False ):

        # ensure parameters are set
        assert dataset_dir is not None

        # set member variables, if trials or sensors are None all will be used
        self._dataset_dir = dataset_dir
        self._trials = trials
        self._sensors = sensors
        self._labels = labels
        self._n_synthetic_sensors_per_label = n_synthetic_sensors_per_label
        self._n_max_samples = n_max_samples
        self._baseline_length = baseline_length

        # derived variables
        self._results_dir = self._dataset_dir
        #'/'.join( self._dataset_dir.split('/')[:-2] ) + '/synthetic_data/'
        
        if not os.path.isdir( self._results_dir ):

            os.makedirs( self._results_dir )

        # figure dir
        self._figure_dir = '/'.join( self._dataset_dir.split('/')[:-2] ) + '/synthetic_data/figures/'
        
        if not os.path.isdir( self._figure_dir ):

            os.makedirs( self._figure_dir )
            
        # save plots
        self._save_plots = save_plots

########################################################################

    def load_data(self):

        """ 
        Read data From pickle files
        """

        # list all pickel files in dataset directory if trials are not specified, otherwise list those selected
        self._pickle_files = os.listdir( self._dataset_dir ) if self._trials is None else [ file for file in os.listdir( self._dataset_dir ) if int( file.split('_')[-1].split('.')[0] ) in self._trials ]

        # read pickle files and generate pandas dataframe
        self._data = pd.DataFrame([pd.read_pickle(self._dataset_dir + '/' + pf ) for pf in self._pickle_files if 'synthetic' not in pf ] )
        
        # keys for generated synthetic sensor data
        self._keys = list(self._data.columns)

        # substring used for naming synthetic data files
        # self._synthetic_file_substring = self._pickle_files[0].split('_')
        # self._synthetic_file_substring.insert(1, 'synthetic')
        # self._synthetic_file_substring = "_".join(self._synthetic_file_substring)
        
########################################################################

    @abstractmethod
    def generate_samples(self):

        """ 
        Abstract method for generating samples
        """

        pass

        
        
########################################################################
