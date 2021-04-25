import numpy as np
from sklearn.model_selection import KFold

import tensorflow as tf

from build_model import build_model
from dataprocessing import PFPGenerator

def train_model(df, meta_column, epochs, data_folder, output_folder):
    kf = KFold(n_splits = 5, random_state = 71, shuffle = True)

    model = build_model(len(meta_column))
    
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
        
        tr_gene = PFPGenerator(df = train, meta = meta_column, batch_size = 32, data_folder = data_folder)
        val_gene = PFPGenerator(df = val, meta = meta_column, batch_size = 32, data_folder = data_folder)
        
        history = model.fit_generator(
            tr_gene,
            steps_per_epoch = 30,
            validation_data = val_gene,
            validation_steps = 16,
            callbacks = [er, cpt, rlp],
            epochs = epochs,
            verbose = 0
        )
        
        tr_eval = model.evaluate(tr_gene)
        val_eval = model.evaluate(val_gene)
        
        print('Train Loss：{:.4f} | Train Score：{:.4f}'.format(tr_eval[0], tr_eval[1]))
        print('Valid Loss：{:.4f} | Valid Score：{:.4f}'.format(val_eval[0], val_eval[1]))