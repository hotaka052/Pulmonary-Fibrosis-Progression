import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

def score(y_true, y_pred):
    """
    評価指標を再現した関数
    """
    c1 = tf.constant(70, dtype = 'float32')
    c2 = tf.constant(1000, dtype = 'float32')
    
    tf.dtypes.cast(y_true, tf.float32)
    tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:,2] - y_pred[:,0]
    fvc_pred = y_pred[:,1]
    
    sigma_clip = tf.maximum(sigma, c1)
    delta = tf.abs(y_true[:,0] - fvc_pred)
    delta = tf.minimum(delta, c2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype = tf.float32))
    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)
    
    return K.mean(metric)

def qloss(y_true, y_pred):
    """
    ピンボールロス
    """
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype = tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q*e, (q-1)*e)
    return K.mean(v)

def mloss(_lambda):
    def loss(y_true, y_pred):
        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)
    return loss