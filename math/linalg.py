# modules
import tensorflow as tf
import math

#######################################
# Compute condition number of matrix
# and check if intertible
# Source: https://stackoverflow.com/questions/57073381/how-to-check-if-a-matrix-is-invertible-in-tensorflow
#######################################

# Based on np.linalg.cond(x, p=None)
def tf_cond(x):

    x = tf.convert_to_tensor(x)
    s = tf.linalg.svd(x, compute_uv=False)
    r = s[..., 0] / s[..., -1]

    # Replace NaNs in r with infinite unless there were NaNs before
    x_nan = tf.reduce_any(tf.is_nan(x), axis=(-2, -1))
    r_nan = tf.is_nan(r)
    r_inf = tf.fill(tf.shape(r), tf.constant(math.inf, r.dtype))

    tf.where(x_nan, r, tf.where(r_nan, r_inf, r))

    return r

def is_invertible(x, epsilon=1e-6):  # Epsilon may be smaller with tf.float64

    x = tf.convert_to_tensor(x)

    eps_inv = tf.cast(1 / epsilon, x.dtype)

    x_cond = tf_cond(x)

    return tf.is_finite(x_cond) & (x_cond < eps_inv)

####################################
# Ensure diagonals of matrix are
# positive.
####################################

def pos_diag(X=None,diag_func=tf.abs):

    """
    Ensure diagnoal values are positive.
    X - matrix whose diagonal values are to be made positive
    diag_func - function to apply to current diagonal values of X, i.e. tf.abs, tf.exp, etc.
    """
    
    assert X is not None

    # extract diagonal from X and make positive
    X_pos_diag = diag_func(tf.diag_part(X))

    # replace diagonal with positive diagonal values
    X = tf.linalg.set_diag(X, X_pos_diag)

    return X
