# dovebirdia
## deeplearning/
    
    networks/ - Feed Forward, Autoencoder, etc.
      
      base.py - AbstractNetwork, FeedForwardNetwork
      autoencoder.py - Autoencoder, AutoencoderKalmanFilter
      lst.py - LSTM
      
## datasets/

    base.py - AbstractDataset
    mnist.py - MNISTDataset
    unpa_op.py - UNPAOPDataset (Based on OpenPose estimates of Facial Features - http://www.unavarra.es/gi4e/databases/hpdb)
    domain_randomization.py - generates training, validation and testing datasets for domain randomization

## filtering/

    base.py - AbstractFilter
    kalman_filter.py - KalmanFilter
    
## utilities/

    base.py - dictToAttributes, saveDict
    distributions.py - statistical distribution functions

## scripts/

    aekf_experiment_generator.py - creates configuration files to run aekf models
    aekf_evaluation_generator.py - creates configuration files to test previously trained aekf models
    dl_model.py - script used to train and test deep learning models
    dr_test_set_generator.py - generates domain randomization test set and saves to disk
    kf_experiment_generator.py - creates configuration files to run Kalman Filter models
    filter_model.py - scripts used to test filter models
