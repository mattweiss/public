from dovebirdia.datasets.base import AbstractDataset

import numpy as np

from pdb import set_trace as st

class FlightSimDataset(AbstractDataset):

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

    initial_slope : float
        intial slope for linear mode

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

    cw : bool (default: True)
        boolearn variable which determines clockwise or counter clockwise turn

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
                 cw,
                 dt,
                 initial_slope,
                 radius_range,
                 angle_range,
                 noise):

        # Passed attributes
        self._n_trials=n_trials
        self._n_samples=n_samples
        self._n_turns=n_turns
        self._cw=cw
        self._dt=dt
        self._slope=initial_slope
        self._radius_range=radius_range
        self._angle_range=angle_range
        self._noise=noise
                
        # Attributes with default values
        self._state='linear'
        self._y=list()
        self._x=list()
        self._x_center=None
        self._y_center=None

        # initial y coordinate
        self._y.append((0.0,0.0))

        # initial dx and dy signs
        self._dx_sign, self._dy_sign = 1.0,np.sign(self._slope)*1.0
        
        # Derived attributes
        self._state_len = self._n_samples // (2*self._n_turns+1)
        
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

                # loop over samples
                for sample in np.arange(self._n_samples)[:]:

                    # check whether to switch state
                    if sample!=0 and (sample+1)%self._state_len==0:

                        # set state
                        self._state = 'turn' if self._state=='linear' else 'linear'

                        # if returning to linear state after initial linear state reset slope
                        if self._state=='linear':

                            self._slope = None
                            self._dx_sign = np.sign(self._y[-1][0]-self._y[-2][0])
                            self._dy_sign = np.sign(self._y[-1][1]-self._y[-2][1])
                            
                        # if returning to turn state reset center of circle
                        if self._state=='turn':

                            self._x_center, self._y_center = None,None
                            
                    # generate sample based on state
                    self._generatePoint(sample)

        # cast y to numpy arrays
        self._y = np.asarray(self._y)

        # generate noisy data
        noise = self._noise[0](**self._noise[1],size=(self._y.shape))
        self._x = self._y + noise
        
        return self._y, self._x, self._x_center, self._y_center
        
    #################
    # Private Methods
    #################

    def _generatePoint(self,sample):

        # linear state
        if self._state=='linear':

            if self._slope is None:

                print(self._dx_sign,self._dy_sign)
                self._slope=-(self._y[-1][0]-self._x_center)/np.sqrt(np.square(self._radius)-np.square(self._y[-1][0]-self._x_center))
                
            # call linear model
            self._y.append(self._linear_model(self._y[-1][0],self._y[-1][1]))

        # turn state
        elif self._state=='turn':

            # sample radius and angle
            self._radius = np.random.uniform(self._radius_range[0],self._radius_range[1])
            self._angle  = np.random.uniform(self._angle_range[0],self._angle_range[1])

            # angle between x-axis and radius vector, relative to positive x-axis
            self._phi = np.pi/2.0 - np.sign(self._slope)*np.arctan(self._slope)
            
            # compute center of circle if both are None
            if self._x_center==None and self._y_center==None:

                # set coefficients of cosine and sine
                if self._slope >= 0.0:

                    self._center_coeff = (-1.0,1.0) if self._cw else (1.0,-1.0)

                else:

                    self._center_coeff = (1.0,1.0) if self._cw else (-1.0,-1.0)

                # compute center of circle
                self._x_center = self._y[-1][0] - self._radius*self._center_coeff[0]*np.cos(self._phi)
                self._y_center = self._y[-1][1] - self._radius*self._center_coeff[1]*np.sin(self._phi)

            self._theta = (np.pi/2.0)/self._state_len #self._radius*self._angle//self._state_len
            self._y.append(self._turn_model(self._y[-1][0],self._y[-1][1],-self._theta))

    def _linear_model(self,x,y):

        lin_x_y = (x+self._dx_sign*self._dt,y+np.abs(self._slope)*self._dy_sign*self._dt)

        return lin_x_y
            
    def _turn_model(self,x,y,theta):

        # counter clockwise turn
        # if not self._cw:

        #     theta = -theta
        
        rot_x_y = self._T(self._x_center,self._y_center)@self._R(theta)@self._T(-self._x_center,-self._y_center)@np.asarray([[x],[y],[1]])

        return rot_x_y[0][0], rot_x_y[1][0]
        
    def _R(self,theta):

        return np.array([[np.cos(theta),-np.sin(theta),0.0],
                         [np.sin(theta),np.cos(theta),0.0],
                         [0.0,0.0,1.0]])
                                  
    def _T(self,x,y):

        return np.array([[1.0,0.0,x],
                         [0.0,1.0,y],
                         [0.0,0.0,1.0]])
