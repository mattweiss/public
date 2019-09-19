import numpy as np
import numpy.polynomial.polynomial as poly
import numpy.polynomial.legendre as leg_poly

def exponential(x,params):

    a,b,c = params
    
    return a * np.exp(b*x) + c

def sigmoid(x,params):

    a,b,c = params
    
    y = a * (1 + np.exp(-b * (x - c)))**-1
    y -= y[0]
    return y

def taylor_poly(x,params):

    return poly.polyval(x,params)

def legendre_poly(x,params):

    return leg_poly.legval(x,params)
