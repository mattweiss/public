from dovebirdia.datasets.flight_sim_dataset import FlightSimDataset

import numpy as np

from pdb import set_trace as st

import matplotlib.pyplot as plt

turns = 2
samples_per_state = 30
samples = (2*turns+1)*samples_per_state

ds = FlightSimDataset(n_trials=1,
                      n_samples=samples,
                      n_turns=turns,
                      cw=True,
                      dt=0.05,
                      initial_slope=-1.0,
                      radius_range=(1.0,1.0),
                      angle_range=(np.pi/4,np.pi/4),
                      noise=(np.random.normal,{'loc':0.0,'scale':1.0}))

y,x,x_c,y_c = ds.getDataset()

#plt.figure(figsize=(6,6))
fig, ax = plt.subplots()
#plt.scatter(x_c,y_c,color='red')

x0 = 0
for x1 in np.arange(samples_per_state,samples+1,samples_per_state):

    plt.scatter(y[x0:x1,0],y[x0:x1,1])
    x0=x1
    
#plt.scatter(y[:,0],y[:,1],color='C0')
# plt.scatter(y[:30,0],y[:30,1],color='C0')
# plt.scatter(y[30:60,0],y[30:60,1],color='C1')
#plt.scatter(y[60:,0],y[60:,1],color='C2')
#ax.add_artist(plt.Circle((x_c,y_c),radius=1.0, color='C4'))
# plt.xlim([-1,5])
# plt.ylim([-1,5])

plt.grid()
plt.show()
plt.close()
