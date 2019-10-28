import os, sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import dill
from pdb import set_trace as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from synthetic_data.synthetic_sensor_data_generator import SyntheticSensorDataGenerator

class PiecewiseSyntheticSensorDataGenerator(SyntheticSensorDataGenerator):

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

        # Setting this by hand for how
        #self._max_height_idx = 1000
        
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
            sensor_labels = list(label_data.sensors.iloc[0])

            # update sensors dictionary with values corresponding to sensor index
            for sensor in self._sensors.keys():

                self._sensors[sensor] = sensor_labels.index(sensor)

            # list of sensor indices for slicing resistance
            sensor_indices = list(self._sensors.values())

            # resistance values for given label, 3 dimensional array (trials, samples, sensors)
            resistance = np.asarray(list(label_data.resistance.values))[:,:,sensor_indices]

            # some resistance arrays are not 3d (?)
            if np.ndim(resistance) == 3:
                
                resistance = resistance[:,:self._n_max_samples,:]

            else:

                continue

            # standardize resistance to baseline
            resistance = np.asarray(list(map(self._baseline_shift,resistance)))
            
            # Piecewise Curve Fit
            params, piecewise_func, trial_max_idx_dict = self._fitPiecewiseModel(resistance)

            # average trial max. response indices for each sensor
            trial_max_idx_mean_dict = dict()
            for sensor, max_res in trial_max_idx_dict.items():

                trial_max_idx_mean_dict[sensor] = int(np.asarray(max_res).mean())
            
            # max and min piecewise parameters
            params_min = np.min(params, axis=0)
            params_max = np.max(params, axis=0)

            # arrays of mean and std for each sample, all sensors
            resistance_std = resistance.std(axis=0)

            # average of standardized baseline standard deviation for each sensor response for given label
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
                ssd_dict['resistance_z'] = np.full((resistance.shape[1],resistance.shape[-1]),np.nan)

                # synthetic curve domain
                t = np.linspace(0,resistance.shape[1],resistance.shape[1])
                
                # loop over sensors
                for idx, sensor in enumerate(self._sensors.keys()):

                    print(idx, sensor)
                    
                    param_list_syn = list()
                    
                    for param_low, param_high in zip(params_min[:,idx],params_max[:,idx]):

                        param_list_syn.append(np.random.uniform(low=param_low, high=param_high))

                    # fit piecewise function with current parameters
                    self._x_hat = trial_max_idx_mean_dict[sensor]
                    y = piecewise_func(t, *param_list_syn)

                    # Clip value in decay region that are larger than maximum in absorbtion region
                    abs_max = np.max(y[:self._x_hat])
                    y[np.where(y>abs_max)] = abs_max
                    
                    # Gaussian draw for synthetic sample
                    noise = np.random.normal(loc=0.0, scale=mean_baseline_std_resistance[idx], size=y.shape)

                    # center noise as np.random.normal results has large mean relative to 0.0
                    noise = noise-noise.mean(axis=0)

                    # insert synthetic sample into synthetic resistance array
                    ssd_dict['resistance_z'][:,idx] = y + noise

                # save synthetic data
                synthetic_data_filename = label_data.iloc[0]['csv_file'].split('_')[0].replace('&','-') + '_synthetic_label_{label}_{syn_ctr}'.format(label=label, syn_ctr=str(synthetic_sensor_ctr))

                with open(self._results_dir + synthetic_data_filename + '.pkl', 'wb') as handle:

                    dill.dump(ssd_dict, handle, protocol=dill.HIGHEST_PROTOCOL)

                synthetic_sensor_ctr += 1

                # save plots
                if self._save_plots:

                    syn_label_ctr = 0
                    
                    for sensor_idx, sensor in enumerate(self._sensors.keys()):
                        
                        fig = plt.figure(figsize=(12,6))

                        plt.suptitle('Label {label}, {sensor}'.format(label=label, sensor=sensor))
                        
                        # real data
                        ax1 = fig.add_subplot(121)

                        trial_list = label_data.csv_file.values
                        
                        for trial_resistance in range(resistance.shape[0]):

                            trial = int(trial_list[trial_resistance].split('_')[-1])
                            
                            ax1.plot(resistance[trial_resistance,:,sensor_idx], label = trial )
                            syn_label_ctr = 1

                        ax1.plot(range(ssd_dict['resistance_z'][:,sensor_idx].shape[0]), ssd_dict['resistance_z'][:,sensor_idx], label='Synthetic', color='black', zorder=10 )
                        ax1.set_xlabel('Sample')
                        ax1.set_ylabel('Standardized Resistance')
                        ax1.set_title('Real Sensor Response with Synthetic Overlay')
                        plt.legend()
                        plt.grid()
                        
                        # synthetic data
                        ax2 = fig.add_subplot(122, sharey=ax1)
                        ax2.plot(ssd_dict['resistance_z'][:,sensor_idx], label=None, color='black', zorder=10 )
                        ax2.set_xlabel('Sample')
                        ax2.set_ylabel('Standardized Resistance')
                        ax2.set_title('Synthetic Sensor Response')
                        plt.grid()

                        #plt.show()
                        plt.savefig(self._figure_dir + synthetic_data_filename + '_{sensor}'.format(sensor=sensor).replace(' ','_'))
                        plt.close()
                    
    def _baseline_shift( self, data ):

        # shift each sensor by mean baseline
        scaler = StandardScaler(with_std = False).fit(data[:self._baseline_length,:])
        return scaler.transform(data)

    def _fitPiecewiseModel(self, data=None):

        assert data is not None
        
        param_array = np.empty((data.shape[0],6,data.shape[-1]))
        piecewise_func_list = list()

        # hold max. response indices for each sensor
        trial_max_idx_dict = dict()
        for sensor in self._sensors.keys():

            trial_max_idx_dict[sensor] = list()
        
        for trial, resistance in enumerate(data):

            #for sensor in range(resistance.shape[-1]):
            for sensor, sensor_idx in self._sensors.items():
                
                sensor_resistance = resistance[:,sensor_idx]
                
                ###############
                # curve fitting
                ###############

                xdata = np.linspace(0,sensor_resistance.shape[0],sensor_resistance.shape[0])
                ydata = sensor_resistance

                max_idx = np.where(sensor_resistance==np.max(sensor_resistance))[0][0]
                self._x_hat = max_idx
                trial_max_idx_dict[sensor].append(max_idx)
                
                try:

                    popt, pcov = curve_fit(self._piecewise_sig, xdata, ydata, p0=[sensor_resistance[self._x_hat],0.1,0.0,10,0.1,0.0], maxfev=20000)
                    piecewise_func = self._piecewise_sig

                except:

                    # popt, pcov = curve_fit(self._piecewise_exp, xdata, ydata, p0=[sensor_resistance[max_idx]/10,0.1,0.0,10,0.1,0.0])
                    # piecewise_func = self._piecewise_exp
                    print('Index:{trial}, Sensor:{sensor} Did Not Converge'.format(trial=trial,sensor=sensor))
                    sys.exit(1)
                    
                param_array[trial,:,sensor_idx] = popt
                piecewise_func_list.append(piecewise_func)

        return param_array, piecewise_func, trial_max_idx_dict

    def _piecewise_sig(self, x,
                       alpha_abs,beta_abs,gamma_abs,
                       alpha_des,beta_des,gamma_des):

        y = np.piecewise(x, 
                            [x <= self._x_hat], 
                            [
                             lambda x: alpha_abs / (1 + np.exp(-beta_abs*(x+gamma_abs))),  # sigmoid
                             lambda x: alpha_des * np.exp(-beta_des*(x-self._x_hat)) + gamma_des  # exponential decay
                            ])

        return y

    def _piecewise_exp(self, x,
                       alpha_abs,beta_abs,gamma_abs,
                       alpha_des,beta_des,gamma_des):
    
        return np.piecewise(x, 
                            [x <= self._x_hat],
                            #[x <= self._max_height_idx],
                            [
                             lambda x: alpha_abs * np.exp(beta_abs*(x)) + gamma_abs, # exponential growth
                             lambda x: alpha_des * np.exp(-beta_des*(x-self._x_hat)) + gamma_des  # exponential decay
                            ])
    
