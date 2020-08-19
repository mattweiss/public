# dovebirdia
## deeplearning/
    
    activations/ - Custom activation fuctions
    
        base.py

    layers/ - Network Layer Classes
    
        base.py - Dense layer class
        
    networks/ - Feed Forward, Autoencoder, etc.
      
        base.py - AbstractNetwork, FeedForwardNetwork
        autoencoder.py - Autoencoder, AutoencoderKalmanFilter
        lstm.py - Keras LSTM
        lstm_tf.py - Tensorflow LSTM
      
      
## datasets/

    base.py - AbstractDataset
    mnist.py - MNISTDataset
    unpa_op.py - UNPAOPDataset (Based on OpenPose estimates of Facial Features - http://www.unavarra.es/gi4e/databases/hpdb)
    domain_randomization.py - generates training, validation and testing datasets for domain randomization

## filtering/

    base.py - AbstractFilter
    distributions.py - Statistical Distributions
    kalman_filter.py - KalmanFilter
    interacting_multiple_model.py - Interacting Multiple Model Estimator
    
## math/

    distributions.py - statistical distribution functions
    divergences.py - various divegences
    metrics.py - various distance metrics for matricies
    linalg.py - linear algebra functions
    
## utilities/

    base.py - dictToAttributes, saveDict
    dr_fuctions.py - Functions used to generate curve families during domain randomization

## scripts/

    a variety of scripts used to generate experiments and scripts to run experiments
    In general, *generator*.py files generate experiments and *model*.py scripts run experiments
    For example, aekf_experiment_generator_dr.py generates AEKF domain randomization experiments and 
    dl_model.py then runs these experiments.
    
## uml/

    uml diagrams for dovebirdia
    
## synthetic_data/

    absorption_synthetic_data_generator.py - Produce synthetic sensor data using unique Gaussian draw at each sample in baseline curve
    homotopy_synthetic_data_generator.py - Produce synthetic sensor data using homotopy interpolation between 2 curves in each label group
    piecewise_synthetic_sensor_data_generator.py - synthetic sensor data fit with parameters fit by piecewise, non-linear regression
    synthetic_sensor_data_generator.py - synthetic sensor data base class
    
