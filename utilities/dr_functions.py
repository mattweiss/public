import numpy as np
import numpy.polynomial.polynomial as poly
import numpy.polynomial.legendre as leg_poly

def exponential(x,a,b,c):

    return a * np.exp(b*x) + c

def sigmoid(x,a,b,c):

    y = a * (1 + np.exp(-b * (x - c)))**-1
    y -= y[0]
    return y

def taylor_poly(x,c):

    return poly.polyval(x,c)

def legendre_poly(x,c):

    return leg_poly.legval(x,c)
