#!/usr/bin/env python
# coding: utf-8

#################
# Modules
#################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from collections import OrderedDict
from pdb import set_trace as st

#################
# Variables
#################

step = 1
SAVEFIGS = True

#################
# Data
#################

results_dir_base = '/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/imm/'
results_file = 'benchmark.pkl'

results_dirs = OrderedDict(
    [
        ('kf_results_dir',('kf_benchmark',1)),
        ('aekf_results_dir',('aekf_turns_1_gaussian_0_20_F_NCV_Q_0-5',18)),
        ('imm_results_dir',('imm_benchmark',1)),
        ('aeimm_results_dir',('aeimm_turns_1_gaussian_0_20_F_NCV_Q_0-5',5))
    ],
)

results = OrderedDict()
for result_key,result_dir in results_dirs.items():
    
    result_data = np.load(results_dir_base+result_dir[0]+'/'+result_dir[0]+'_model_{model}'.format(model=result_dir[1])+'/results/'+results_file,allow_pickle=True)

    if 'aekf' in result_key or 'aeimm' in result_key:

        results[result_key] = {
                                'z':result_data['x'],
                                'z_hat':result_data['y_hat'],
                                'z_true':result_data['y'],
                                'R':np.asarray([kf_res['R'] for kf_res in result_data['kf_results']])
        }

    elif 'lstm' in result_key:

        results[result_key] = {
                                'z':result_data['x'][:,:,0],
                                'z_hat':result_data['y_hat'],
                                'z_true':result_data['y'],
        }
        
    else:
        
        results[result_key] = result_data

for k,v in results.items():
    
    print(k,v.keys())

#################
# Parameters
#################

n_result_trials = results[list(results.keys())[0]]['z_true'].shape[0]
n_tests = len(results.keys())

#################
# Plot
#################

# loop over each trial
for trial in range(n_result_trials)[12:13]:
    
    # create figure
    size = 6
    w = 2 * size
    h = 2 * size
    fig, ax = plt.subplots(n_tests//2, n_tests//2, figsize=(w,h))
    plt.subplots_adjust(hspace=0.25)
    
    if np.ndim(ax) == 1:
        
        ax = np.expand_dims(ax,axis=0)

    # dictionary for all trial mse values
    mse_dict = OrderedDict()

    row = col = 0
    
    # loop over tests
    for result_key_index, result_key in enumerate(results.keys()):

        # test name
        test = result_key.split('_')[0]
                
        # data
        z = results[result_key]['z'][trial]
        z_true = results[result_key]['z_true'][trial]
        z_hat = results[result_key]['z_hat'][trial]

        # mse for current result
        mse_dict[result_key] = np.square(np.subtract(z_true,z_hat)).mean()

        # z, z true, z hat
        ax[row,col].scatter(z[:,0],z[:,1],label='Noisy Measurements',marker='o',s=5,color='C0')
        ax[row,col].plot(z_true[::step,0],z_true[::step,1],label='Ground Truth',color='C2')
        ax[row,col].plot(z_hat[::step,0],z_hat[::step,1],label='State Estimate',color='C3',linewidth=3)
        mse=np.sqrt(mse_dict[result_key])
        mse_ratio = mse / np.sqrt(mse_dict['kf_results_dir'])
        #ax[row,col].text(480,300,'RMSE {mse:0.2f} ({mse_ratio:0.2f})'.format(mse=mse,mse_ratio=mse_ratio),fontweight='bold')
        ax[row,col].text(810,-150,'RMSE {mse:0.2f} ({mse_ratio:0.2f})'.format(mse=mse,mse_ratio=mse_ratio),fontweight='bold')
        ax[row,col].grid()
        ax[row,col].legend()
        ax[row,col].set_title('{test}'.format(test=test.upper()),fontweight=None)
        ax[row,col].set_xlabel('x (m)')
        ax[row,col].set_ylabel('y (m)')
        ax[row,col].axis('equal')
        ax[row,col].set_xlim(500,1300)
        ax[row,col].set_ylim(top=100,bottom=-600)
                
        if (result_key_index+1)%2==0:

            row+=1
            col=0

        else:
            
            col+=1

    # suptitle
    suptitle_str = ''
    # for k,v in mse_dict.items():
        
    #     suptitle_str += '{test} {test_mse:0.2e}, '.format(test=k.split('_')[0],test_mse=v)
        
    # imm/kf mse ratio
    mse_values = np.asarray(list(mse_dict.values()))
    mse_ratios = (mse_values / mse_values[0])

    #suptitle_str += '\nMSE Ratios: {mse_ratios}'.format(mse_ratios=np.around(mse_ratios,3))
    #plt.suptitle(suptitle_str)

    if SAVEFIGS:

        plt.savefig('./experiments/{res_dir}/figs/{trial}'.format(res_dir=results_dir_base.split('/')[-2],
                                                                  trial=trial),dpi=300)

    else:
        
        plt.show()

    plt.close()

