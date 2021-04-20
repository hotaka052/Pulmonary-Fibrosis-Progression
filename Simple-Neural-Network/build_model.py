import tensorflow_addons as tfa
import tensorflow.keras.layers as L
from tensorflow.keras import Model

from losses import mloss, score

def build_model(n_meta):
    input = L.Input(shape = (n_meta, ))
    x = L.Dense(200, kernel_initializer = 'he_normal')(input)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(alpha = 0.1)(x)
    x = L.Dense(200, kernel_initializer = 'he_normal')(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU(alpha = 0.1)(x)
    output = L.Dense(3, kernel_initializer = 'he_normal')(x)
    
    model = Model(input, output)
    
    model.compile(optimizer = tfa.optimizers.RectifiedAdam(learning_rate = 0.05), loss = mloss(0.8), metrics = score)
    
    return model