import tensorflow as tf
from dovebirdia.utilities.base import dictToAttributes

def orthonormal_regularizer(scale, scope=None):

    def ortho_reg(weights):

        with tf.name_scope(scope, 'orthonormal_regularizer', [weights]) as name:

            norm = tf.norm(tf.matmul(weights,weights,transpose_a=True) - tf.eye(tf.shape(weights)[1],dtype=tf.float64))
        
            my_scale = tf.convert_to_tensor(scale,
                                            dtype=weights.dtype.base_dtype,
                                            name='scale')
        
            return tf.multiply(my_scale, norm, name=name)

    return ortho_reg
