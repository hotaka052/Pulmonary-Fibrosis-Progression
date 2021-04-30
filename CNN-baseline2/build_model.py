import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as L
import tensorflow.keras.applications.efficientnet as efn
from tensorflow.keras import Model

def build_model(n_meta, shape = (256, 256, 3)):
    #基本モデル
    model = efn.EfficientNetB4(weights = 'imagenet', input_shape = shape, include_top = False)

    #画像データのネットワーク
    inp1 = L.Input(shape = shape)
    x = model(inp1)
    x = L.Flatten()(x)
    x = L.Dense(100)(x)
    x = L.LeakyReLU(alpha = 0.1)(x)
    
    #テーブルデータのネットワーク
    inp2 = L.Input(shape = (n_meta,))
    x2 = L.Dense(500)(inp2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(alpha = 0.1)(x2)
    x2 = L.Dropout(rate = 0.5)(x2)
    x2 = L.Dense(100)(x2)
    x2 = L.BatchNormalization()(x2)
    x2 = L.LeakyReLU(alpha = 0.1)(x2)
    x2 = L.Dropout(rate = 0.5)(x2)
    
    #合流
    added = L.Concatenate()([x,x2])
    output = L.Dense(1)(added)
    
    model = Model([inp1, inp2], output)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = 'mae')
    
    return model