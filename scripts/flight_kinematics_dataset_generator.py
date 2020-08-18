from dovebirdia.datasets.flight_kinematics import FlightKinematicsDataset

import numpy as np
from numpy import pi as PI

from pdb import set_trace as st

import matplotlib.pyplot as plt

turns = 1
samples_per_state = 100
samples = (2*turns+1)*samples_per_state
n_trials=100

ds = FlightKinematicsDataset(n_trials=n_trials,
                             n_samples=samples,
                             n_turns=turns,
                             dt=0.1,
                             r0=(0.0,0.0),
                             v0=(100.0,0.0),
                             radius_range=(200.0,300.0),
                             angle_range=(np.pi/4,np.pi/4),
                             cw=1,
                             noise=(np.random.normal,{'loc':0.0,'scale':20}),
                             save_path='/home/mlweiss/Documents/wpi/research/code/dovebirdia/experiments/dissertation/imm/eval/benchmark.pkl'
)

data = ds.generateDataset()

if data is not None:
    
    for trial in np.arange(n_trials):

        r,r_n,v = data['y_test'][trial],\
                  data['x_test'][trial],\
                  data['vy_test'][trial]

        fig, ax = plt.subplots(1,1,figsize=(8,8))

        sample_splits = np.array_split(np.arange(samples),2*turns+1)

        # plt.text(r[sample_splits[0][0],0],r[sample_splits[0][0],1],'Start',fontsize=15)
        # plt.text(r[sample_splits[-1][-1],0],r[sample_splits[-1][-1],1],'End',fontsize=15)

        for split in sample_splits[:]:

            #ax.scatter(center[0],center[1],color='red')

            ax.plot(r[split,0],r[split,1])
            ax.scatter(r_n[split,0],r_n[split,1],s=5)

            #speed = 25*(v[split,0])**2+(v[split,1])**2
            #ax.arrow(r[split,0],r[split,1],v[split,0]/speed,v[split,1]/speed,width=0.005)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid()
        ax.axis('equal')
        #plt.savefig('{trial}.png'.format(trial=trial), dpi=300)
        plt.show()
        plt.close()
