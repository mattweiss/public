# modules
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import logm
from pdb import set_trace as st

############################################
# Affine-Invariant Distance for SPD matrices
#
# Riemannian Metric Learning for Symmetric
# Positive Definite Matrices
# Raviteja Vemulapalli, David W. Jacobs
############################################

def affineInvariantDistance(P1=None,P2=None):

    assert P1 is not None
    assert P2 is not None
    
    """
    P1 and P2 are two symmetric positive-definite matrices
    """

    # sqrt-inverse of P1
    P1_inv_sqrt = sqrtm(np.linalg.inv(P1))

    # distance calculation
    d = np.linalg.norm(logm(P1_inv_sqrt @ P2 @ P1_inv_sqrt),ord='fro')

    return d

############################################
# Log Frobenius Distance for SPD matrices
#
# Riemannian Metric Learning for Symmetric
# Positive Definite Matrices
# Raviteja Vemulapalli, David W. Jacobs
############################################

def logFrobeniusDistance(P1=None,P2=None):

    """
    P1 and P2 are two symmetric positive-definite matrices
    """
    
    assert P1 is not None
    assert P2 is not None
    
    # distance
    d = np.linalg.norm((logm(P1) - logm(P2)),ord='fro')

    return d
