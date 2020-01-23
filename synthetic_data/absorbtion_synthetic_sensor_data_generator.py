import os, sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import dill
from pdb import set_trace as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from synthetic_data.synthetic_sensor_data_generator import SyntheticSensorDataGenerator

class AbsorbtionSyntheticSensorDataGenerator(SyntheticSensorDataGenerator):

    """ 
    Produce synthetic sensor data using unique Gaussian draw at each sample in baseline curve
    baseline curve - piecewise non-linear least squares, where parameters are determined by label-group samples
    mean - average of all curves in label-group
    std  - average baseline std of all curves in label-group
    """

    def __init__( self,
                  dataset_dir = None,
                  trials = None,
                  sensors = None,
                  labels = None,
                  n_synthetic_sensors_per_label = 1,
                  use_baseline_std = False,
                  save_plots = False):

        super().__init__(dataset_dir = dataset_dir,
                         trials = trials,
                         sensors = sensors,
                         labels = labels,
                         n_synthetic_sensors_per_label = n_synthetic_sensors_per_label,
                         save_plots = save_plots )

        # slice of samples to use - HARDCODE FOR NOW
        self._sample_slice = (600,1000)

        # sampled frequency
        self._sample_freq = 20.0
        
    def generate_samples(self):

        """ 
        Piecewise non-linear least squares curve fitting to determine baseline curves.
        Add Gaussian noise parameterized by (a) mean of curves in label-group and (b) mean baseline std of label-group
        """

        # group by class label
        self._labels = [ label for label in self._data.label.unique() if label in self._labels ] if self._labels is not None else self._data.label.unique()

        # synthetic sensor counter.  Used for naming synthetic sensor data dill files
        synthetic_sensor_ctr = 1
        
        # loop over labels
        for label in self._labels:

            # trials for given label
            label_data = self._data[self._data.label==label]

            # names of sensors
            sensor_names = list(label_data.sensors.iloc[0])

            # update sensors dictionary with values corresponding to sensor index
            # initiall the values in self._sensors are -1 because not all sensors will necessarily be used
            # to ensure the used sensors are labeled accordingly the below code is needed
            for sensor in self._sensors.keys():

                self._sensors[sensor] = sensor_names.index(sensor)

            # resistance values for given label, 3 dimensional array (trials, samples, sexnsors)
            resistance = np.asarray(list(label_data.resistance.values))
            
            # some resistance arrays are not 3d
            if np.ndim(resistance) == 3:
                
                resistance = resistance[:,:self._n_max_samples,:]

            else:

                continue

            # baseline shift
            resistance = np.asarray(list(map(self._preprocess,resistance)))

            # select subset of data
            resistance = resistance[:,self._sample_slice[0]:self._sample_slice[1],:]
            
            # fit curve model
            params, piecewise_func, trial_max_idx_dict = self._fitCurveModel(resistance)

            # average trial max. response indices for each sensor
            #trial_max_idx_mean_dict = dict()
            
            # max and min estimated parameters for curve model
            params_min = np.nanmin(params, axis=0)
            params_max = np.nanmax(params, axis=0)

            st()
            
            # arrays of mean and std for each sample, all sensors
            # resistance_std = resistance.std(axis=0)
            
            # average baseline standard deviation for each sensor response
            # this is the standard deviation of the sampled noise added to the synthetic baseline curve
            mean_baseline_std_resistance = resistance[:,:self._baseline_length,: ].std(axis=1).mean(axis=0)
            
            # generate synthetic sensor data
            for _ in range(self._n_synthetic_sensors_per_label): 

                # dictionary for synthetic sensor data
                ssd_dict = dict.fromkeys(self._keys)
                
                # populate dictionary values
                ssd_dict['synthetic'] = True
                ssd_dict['label'] = label
                ssd_dict['concentration'] = label_data.iloc[0].concentration
                ssd_dict['event_indices'] = label_data.iloc[0].event_indices
                ssd_dict['name'] = label_data.iloc[0].name
                ssd_dict['sensors'] = list(self._sensors.keys())
                ssd_dict['time'] = label_data.iloc[0].time[:self._n_max_samples]
                ssd_dict['y'] = label_data.iloc[0].y
                ssd_dict['y_multi_label'] = label_data.iloc[0].y

                # array to hold resistance
                ssd_dict['resistance'] = np.full((resistance.shape[1],resistance.shape[-1]),np.nan)

                # synthetic curve domain
                t = np.linspace(0,resistance.shape[1],resistance.shape[1]) / int(self._sample_freq)

                # loop over sensors
                for sensor, sensor_idx in self._sensors.items():

                    # list of parameters used in synthetic curve genration
                    param_list_syn = list()

                    # for each parameter in curve model, sample uniformly between the min and max values
                    # determined by curve fitting above
                    for param_low, param_high in zip(params_min[:,sensor_idx],params_max[:,sensor_idx]):

                        try:

                            param_list_syn.append(np.random.uniform(low=param_low, high=param_high))

                        except:

                            print(param_low, param_high)

                    # generate ground truth synthetic curve
                    print(*param_list_syn)
                    y = piecewise_func(t, *param_list_syn)

                    # Gaussian draw for synthetic sample
                    noise = np.random.normal(loc=0.0, scale=mean_baseline_std_resistance[sensor_idx], size=y.shape)

                    # center noise as np.random.normal results has large mean relative to 0.0
                    noise = noise-noise.mean(axis=0)

                    # insert synthetic sample into synthetic resistance array
                    ssd_dict['resistance'][:,sensor_idx] = y + noise

                # save synthetic data
                synthetic_data_filename = label_data.iloc[0]['csv_file'].split('_')[0].replace('&','-') + '_synthetic_label_{label}_{syn_ctr}'.format(label=label, syn_ctr=str(synthetic_sensor_ctr))

                with open(self._results_dir + synthetic_data_filename + '.pkl', 'wb') as handle:

                    dill.dump(ssd_dict, handle, protocol=dill.HIGHEST_PROTOCOL)

                synthetic_sensor_ctr += 1

                # save plots
                if self._save_plots:

                    syn_label_ctr = 0
                    
                    #for sensor_idx, sensor in enumerate(self._sensors.keys()):
                    for sensor, sensor_idx in self._sensors.items():
                        
                        fig = plt.figure(figsize=(12,6))

                        plt.suptitle('Label {label}, {sensor}'.format(label=label, sensor=sensor))
                        
                        # real data
                        ax1 = fig.add_subplot(121)
                        ax2 = fig.add_subplot(122, sharey=None)
                        
                        trial_list = label_data.csv_file.values
                        
                        for trial_resistance in range(resistance.shape[0]):

                            trial = int(trial_list[trial_resistance].split('_')[-1])
                            print(trial)
                            ax1.plot(t,resistance[trial_resistance,:,sensor_idx], label = trial )
                            ax2.plot(t,resistance[trial_resistance,:,sensor_idx], label = trial )
                            syn_label_ctr = 1

                        ax1.plot(t, ssd_dict['resistance'][:,sensor_idx], label='Synthetic', color='black', zorder=10 )
                        ax1.set_xlabel('Sample')
                        ax1.set_ylabel('Standardized Resistance')
                        ax1.set_title('Real Sensor Response with Synthetic Overlay')
                        plt.legend()
                        plt.grid()

                        # synthetic data

                        # ax2.plot(ssd_dict['resistance'][:,sensor_idx], label=None, color='black', zorder=10 )
                        # ax2.set_xlabel('Sample')
                        # ax2.set_ylabel('Standardized Resistance')
                        # ax2.set_title('Synthetic Sensor Response')
                        # plt.grid()

                        plt.show()
                        #plt.savefig(self._figure_dir + synthetic_data_filename + '_{sensor}'.format(sensor=sensor).replace(' ','_'))
                            
                        plt.close()
                    
    def _preprocess( self, data ):

        # min-max scaling
        data = MinMaxScaler(feature_range=(0.0,1)).fit_transform(data)
        
        # baseline shift
        scaler = StandardScaler(with_std = False).fit(data[:self._baseline_length])
        data = scaler.transform(data)

        return data

    def _fitCurveModel(self, data=None):

        assert data is not None

        param_array = np.full((data.shape[0],3,data.shape[-1]),np.nan)
        piecewise_func_list = list()

        # hold max. response indices for each sensor
        trial_max_idx_dict = dict()
        for sensor in self._sensors.keys():
            
            trial_max_idx_dict[sensor] = list()

        for trial, resistance in enumerate(data):

            # sensor counter to populate param_array's 3rd dimension
            sensor_ctr = 0

            #for sensor in range(resistance.shape[-1]):
            for sensor, sensor_idx in self._sensors.items():
                
                sensor_resistance = resistance[:,sensor_idx].astype(np.float64)

                ###############
                # curve fitting
                ###############

                xdata = np.linspace(0,sensor_resistance.shape[0],sensor_resistance.shape[0]) / self._sample_freq
                ydata = sensor_resistance

                # plt.figure(figsize=(6,6))
                # plt.plot(xdata,ydata)
                # plt.grid()
                # plt.show()
                # plt.close()

                # sigmoid
                #try:
                    
                # popt, pcov = curve_fit(self._sig_curve, xdata, ydata, maxfev=10000)
                # piecewise_func = self._sig_curve
                    #print('Sigmoid')

                # exponential
                # except:

                popt, pcov = curve_fit(self._exp_curve, xdata, ydata)
                piecewise_func = self._exp_curve
                #     print('Exponential')

                param_array[trial,:,sensor_idx] = popt
            
        return param_array, piecewise_func, trial_max_idx_dict
    
    def _exp_curve(self,x,a,b,c):

        return a*np.exp(b*(x-c))

    def _sig_curve(self,x,a,b,c):
    
        return a*(1+np.exp(-b*(x-c)))**-1
