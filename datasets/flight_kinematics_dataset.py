from dovebirdia.datasets.base import AbstractDataset

import numpy as np
from numpy import pi as PI

from pdb import set_trace as st

class FlightKinematicsDataset(AbstractDataset):

    """
    Generates simluated 2d flight data with noise


    Attributes
    ----------

    #####################
    # Required attributes
    #####################
    
    n_trials : int
        total number of simulated flight patterns

    n_samples : int
        total number of simulated samples per trial

    n_turns : int
        total number of turns

    dt : float
        amount to increment each sample

    r0 : (float,float)
        initial x and y coordinates
    
    v0 : (float,float)
        initial x and y velocities

    radius_range : (float,float)
        range of radius values for turn mode

    angle_range : (float,float)
        range of turn angle values (radians) for turn mode

    noise : (noise function, noise parameters)
        noise function and dictionary of parameters
        Ex: (np.random.normal, {'loc':0.0,'scale':1.0})
    
    ####################
    # Default attributes
    ####################

    state : string (default: 'linear')
        indicates current state of model (linear or turn)

    y : list()
        simulated ground truth flight path

    x : list()
        simulated ground truth flight path with added noise
    
    """

    def __init__(self,
                 n_trials,
                 n_samples,
                 n_turns,
                 dt,
                 r0,
                 v0,
                 radius_range,
                 angle_range,
                 cw,
                 noise):

        # Passed attributes
        self._n_trials=n_trials
        self._n_samples=n_samples
        self._n_turns=n_turns
        self._dt=dt
        self._r=np.asarray(r0)
        self._v=v0
        self._radius_range=radius_range
        self._angle_range=angle_range
        self._cw=cw
        self._noise=noise

        # Attributes with default values
        self._state='linear'
        self._history=dict()
        self._history['r']=list()
        self._history['v']=list()
        
        # Generate samples for each segment
        self._segments = np.array_split(np.arange(self._n_samples),2*self._n_turns+1)

        # slope of velocity vector
        self._theta = np.arctan(self._v[1]/self._v[0])

        # orientation of turn relative to positive x-axis
        self._phi = None
        
        # Center of turn
        self._center = np.empty(2).fill(np.nan)

    ################
    # Public Methods
    ################

    def getDataset(self,load_path=None):

         # load previously saved dataset
        if load_path is not None:

            self._data = self.getSavedDataset(load_path)

        # generate dataset
        else:

            # loop over trials
            for trial in np.arange(self._n_trials):

                # loop over segments
                for segment_ctr, segment in enumerate(self._segments[:]):

                    if segment_ctr>0:

                        if self._state=='linear':

                            self._state='turn'
                            self._cw=np.random.choice([-1,1])
                            print(self._cw)
                            
                        else:

                            self._state='linear'
                            self._phi=None
                        
                    # update position vector
                    if self._state == 'linear':

                        self._r, self._v = self._linear_model(segment)

                    elif self._state == 'turn':

                        self._r, self._v = self._turn_model(segment)

        # Add noise
        noise = self._noise[0](**self._noise[1],size=np.asarray(self._history['r']).shape)
        self._history['r_noise'] = self._history['r'] + noise
        
        return self._history, self._center
        
    #################
    # Private Methods
    #################

    def _linear_model(self,segment=None):

        assert segment is not None

        print('Linear Model')

        if self._center is not None:
            
            self._center = np.empty(2).fill(np.nan)

        for k in np.arange(len(segment)):

            # position
            if len(self._history['r'])!=0:
                
                self._r = np.array([self._v[0]*self._dt,self._v[1]*self._dt]) + self._r

            # velocity
            self._v = self._v

            self._history['r'].append(self._r)
            self._history['v'].append(self._v)
            
        return self._r,self._v

    def _turn_model(self,segment=None):

        assert segment is not None

        print('Turn Model')

        # list to hold position and velocity values
        # this is necessary since we need to reverse the lists before appending to history
        r_list = list()
        v_list = list()
        
        radius = np.random.uniform(self._radius_range[0],self._radius_range[1])
        initial_slope = self._v[1]/self._v[0]
        speed = np.sqrt(self._v[0]**2+self._v[1]**2)

        if self._center is None:
            
            self._center = np.empty(2)
            self._center[0] = self._r[0] + np.sign(self._v[0])*np.sign(self._cw)*np.divide(initial_slope*radius,np.sqrt(1+initial_slope**2))
            self._center[1] = self._r[1] - np.sign(self._v[0])*np.sign(self._cw)*np.sqrt(radius**2-(self._r[0]-self._center[0])**2)

        theta = speed*self._dt*len(segment)/radius

        if self._phi is None:

            if self._cw==1:

                if self._v[0]>0.0:
                
                    self._r_phi = -theta+PI/2+np.arctan(initial_slope)

                else:

                    self._r_phi = -theta+3*PI/2+np.arctan(initial_slope)
                    
                self._v_phi = PI
                
            elif self._cw==-1:

                if self._v[0]>0.0:
                
                    self._r_phi = -PI/2+np.arctan(initial_slope)+speed*self._dt/radius

                else:

                    self._r_phi = PI/2+np.arctan(initial_slope)+speed*self._dt/radius
                    
                self._v_phi = 0
                
        for k in np.arange(len(segment)):

            # turn
            r = np.asarray([radius*np.cos(speed*k*self._dt/radius),radius*np.sin(speed*k*self._dt/radius)])

            v = np.asarray([-speed*np.sin(speed*k*self._dt/radius), speed*np.cos(speed*k*self._dt/radius)])

            r_list.append(r)
            v_list.append(v)
            
        # rotate and translate
        for k in np.arange(len(segment)):
            pass
            # position
            r_list[k] = self._Rot(self._r_phi)@r_list[k]
            r_list[k] = r_list[k]+self._center

            # # velocity
            v_list[k] = self._Rot(self._r_phi)@self._Rot(self._v_phi)@v_list[k]
            
        # reverse position and velocity lists and append to history

        if self._cw==1:

            r_list.reverse()
            v_list.reverse()

        self._history['r'] = self._history['r'] + r_list
        self._history['v'] = self._history['v'] + v_list

        return self._history['r'][-1],self._history['v'][-1]

    def _Rot(self,theta):

        return np.array([[np.cos(theta),-np.sin(theta)],
                         [np.sin(theta),np.cos(theta)]])
