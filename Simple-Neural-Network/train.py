import numpy as np
from sklearn.model_selection import KFold

import tensorflow as tf

from build_model import build_model
from data_processing import PFPGenerator

def train_model(df, meta, epochs, output_folder):
    kf = KFold(n_splits = 5, random_state = 37, shuffle = True)

    model = build_model(len(meta))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(df)):
        print('=' * 10, 'Fold', fold, '=' * 10)
        
        train = df.iloc[tr_idx]
        val = df.iloc[val_idx]
        
        #early_stopping
        er = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 10,
            mode = 'auto'
        )
    
        #モデルを保存
        cpt = tf.keras.callbacks.ModelCheckpoint(
            filepath = output_folder + f'/weight_fold{fold}.h5',
            monitor = 'val_loss',
            save_best_only = True,
            mode = 'auto'
        )
    
        #学習が進まなくなったらLearning_rateを下げる
        rlp = tf.keras.callbacks.ReduceLROnPlateau(
            monitor = 'val_loss',
            factor = 0.5,
            patience = 3,
            verbose = 0
        )
        
        tr_gene = PFPGenerator(df = train, meta = meta, batch_size = 128)
        val_gene = PFPGenerator(df = val, meta = meta, batch_size = 128)
        
        history = model.fit_generator(
            tr_gene,
            steps_per_epoch = 30,
            validation_data = val_gene,
            validation_steps = 16,
            callbacks = [er, cpt, rlp],
            epochs = epochs,
            verbose = 0
        )
        
        x_tr = np.array(train[meta].values, dtype = np.float32)
        y_tr = np.array(train['FVC'].values, dtype = np.float32)
        x_val = np.array(val[meta].values, dtype = np.float32)
        y_val = np.array(val['FVC'].values, dtype = np.float32)
        
        tr_eval = model.evaluate(x_tr, y_tr, verbose = 0, batch_size = 128)
        val_eval = model.evaluate(x_val, y_val, verbose = 0, batch_size = 128)
        
        print('Train Loss：{:.4f} | Train Score：{:.4f}'.format(tr_eval[0], tr_eval[1]))
        print('Valid Loss：{:.4f} | Valid Score：{:.4f}'.format(val_eval[0], val_eval[1]))