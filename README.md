# dovebirdia
## deeplearning/
    
    networks/ - Feed Forward, Autoencoder, etc.
      
      base.py - AbstractNetwork, FeedForwardNetwork
      autoencoder.py - Autoencoder, AutoencoderKalmanFilter
      
## datasets/

    base.py - AbstractDataset
    mnist.py - MNISTDataset
    unpa_op.py - UNPAOPDataset (Based on OpenPose estimates of Facial Features - http://www.unavarra.es/gi4e/databases/hpdb)
    domain_randomization.py - Generates training, validation and testing datasets for domain randomization

## filtering/

    base.py - AbstractFilter
    kalman_filter.py - KalmanFilter
    
## utilities/

    base.py - dictToAttributes, saveDict

## scripts/

    various scripts related to testing code
