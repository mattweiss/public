import tensorflow as tf

try:
    
    from orthnet import Legendre, Chebyshev

except:

    pass

def sineline(x):

    with tf.variable_scope('sineline'):

        alpha = tf.constant(1e-0, shape=[x.get_shape()[-1]], dtype=tf.float64, name='alpha')

    # beta  = tf.constant(1e-1, shape=[x.get_shape()[-1]])
    # gamma  = tf.constant(1e-1, shape=[x.get_shape()[-1]])

    return x + alpha * tf.sin(x)

def psineline(x):

    with tf.variable_scope('psineline'):

        alpha = tf.Variable(tf.constant(0.0, shape=[x.get_shape()[-1]], dtype=tf.float64), name='alpha')
        beta  = tf.Variable(tf.constant(1.0, shape=[x.get_shape()[-1]], dtype=tf.float64), name='beta')
        gamma = tf.Variable(tf.constant(0.0, shape=[x.get_shape()[-1]], dtype=tf.float64), name='gamma')

        # alpha = tf.Variable(tf.constant(0.0, shape=[x.get_shape()[-1]]),  dtype=x.dtype, name='alpha', constraint=lambda t: tf.clip_by_value(t, 0, 1) )
        # beta  = tf.Variable(tf.constant(1.0, shape=[x.get_shape()[-1]]),  dtype=x.dtype, name='beta',  constraint=lambda t: tf.clip_by_value(t, 0, 1))
        # gamma = tf.Variable(tf.constant(0.0, shape=[x.get_shape()[-1]]),  dtype=x.dtype, name='gamma', constraint=lambda t: tf.clip_by_value(t, 0, 1))

    return x + alpha * tf.sin(x)

def tanhpoly(x):

    N=3
    c = tf.get_variable(name='c_tanhpoly',
                        shape=(N+1,x.get_shape()[-1]),
                        initializer=tf.initializers.glorot_uniform,
                        trainable=True,
                        #constraint=lambda t: tf.clip_by_value(t, -1, 1),
                        dtype=tf.float64)

    P = Legendre(x, N).tensor
    x = tf.matmul(P,c)
        
    return x
