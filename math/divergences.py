# modules
import numpy as np
from scipy.linalg import logm
from pdb import set_trace as st

##########################################
# KL-Divergence for multivariate Gaussians
##########################################

def KLDivergence(N0,N1):

    """
    N0, and N1 are tuples with stucture (loc,scale)
    """
    
    N0_mean, N0_cov = N0
    N1_mean, N1_cov = N1

    # univariate Gaussian
    if np.isscalar(N0_mean):
        
        kl_div = np.log(np.sqrt(N1_cov)/np.sqrt(N0_cov)) + (N0_cov+(N0_mean-N1_mean)**2)/(2*N1_cov) - 0.5
    
    # multivariate Gaussian
    else:
        
        N1_cov_inv = np.linalg.inv(N1_cov)
        N1_cov_det = np.linalg.det(N1_cov)
        N0_cov_det = np.linalg.det(N0_cov)
        k = N0_mean.shape[0]
        
        kl_div = 0.5*(np.trace(N1_cov_inv@N0_cov) + (N1_mean-N0_mean).T@N1_cov_inv@(N1_mean-N0_mean)
                      - k + np.log(N1_cov_det/N0_cov_det) )

    return kl_div

##########################################
# von Neumann Entropy
# Source: Information Geometry and its
# Applications, Shun-ichi Amari, pg. 12
##########################################

def vonNeumannEntropyDivergence(P1,P2):

    """
    P1 and P2 are two positive definite matrices
    """

    assert P1 is not None
    assert P2 is not None
    
    div = np.trace(P1 @ logm(P1) - P1 @ logm(P2) - P1 + P2)

    return div

##########################################
# SPD KL Divergence-MVG Derivative
# Source: Information Geometry and its
# Applications, Shun-ichi Amari, pg. 12
##########################################

def logDetDivergence(P1,P2):

    """
    P1 and P2 are two positive definite matrices
    """

    assert P1 is not None
    assert P2 is not None

    P1_dot_P2_inv = P1 @ np.linalg.inv(P2)
    
    div = np.trace(P1_dot_P2_inv) - np.log( np.linalg.det(P1_dot_P2_inv) ) - P1.shape[0]

    return div
