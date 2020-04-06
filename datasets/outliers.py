import numpy as np
from pdb import set_trace as st

def generate_outliers(shape=None,
                      p_outlier=None,
                      outlier_range=(0,1)):

    assert shape is not None
    assert p_outlier is not None
    
    # outliers 
    outlier_samples = np.random.uniform(low=outlier_range[0],high=outlier_range[1],size=shape)

    # binary array to subselect outliers
    binary_array = np.random.choice([0,1], size=shape, p=[1-p_outlier,p_outlier])

    # indices of 1 values in binary_array
    binary_keep_indices = np.where(binary_array==1)[0]

    # randomly select half of the binary keep indices, excluding potentially the final index as adding 1 causes error
    binary_keep_indices = np.random.choice(binary_keep_indices[binary_keep_indices!=shape[0]-1],
                                           size=binary_keep_indices.shape[0]//2,
                                           replace=False)

    # ensure at least half of the outliers are sequential
    binary_array[binary_keep_indices+1] = 1

    # elementwise product of outliers and binary array
    train_outliers = outlier_samples * binary_array

    return train_outliers
