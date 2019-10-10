import numpy as np
import numpy.polynomial.polynomial as poly
import numpy.polynomial.legendre as leg_poly
from pdb import set_trace as st

def exponential(x,params):

    assert x is not None
    assert params is not None
    
    a,b,c = params
    
    return a * np.exp(b*x) + c

def sigmoid(x,params):

    assert x is not None
    assert params is not None
    
    a,b,c = params

    y = a * (1 + np.exp(-b * (x - c)))**-1
    y -= y[0]
    return y

def sine(x,params):

    assert x is not None
    assert params is not None
    
    a,b = params

    return a * np.sin(b*x)

def taylor_poly(x,params):

    assert x is not None
    assert params is not None
    
    return poly.polyval(x,params)

def legendre_poly(x,params):

    assert x is not None
    assert params is not None

    return leg_poly.legval(x,params)

def trig_poly(x,params=None):
    
    assert x is not None
    assert params is not None

    a0 = params[0]
    m = len(params[1:])//2 + 1
    a_params = params[1:m]
    b_params = params[m:]
    
    y = a0
    
    for n, (a,b) in enumerate(zip(a_params,b_params)):
        
        y += a*np.cos(n*x) + b*np.sin(n*x)
        
    return y
