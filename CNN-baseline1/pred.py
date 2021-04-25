import numpy as np

from build_model import build_model

from dataprocessing.processing_generator import *

def get_item(df, meta_column, data_folder):
    """
    推論データを用意する関数
    """
    x, meta = [], []
    
    for i in range(len(df)):
        row = full_path(df.iloc[i], phase = 'test', data_folder = data_folder)
        img = get_img(row['full_path'])
        img = mono_to_color(img)
        img = img / 255.0
        x.append(img)
        
        meta.append(df.iloc[i][meta_column])
        
    x = np.array(x, dtype = np.float32)
    meta = np.array(meta, dtype = np.float32)
    
    return [x, meta]

def predict(df, meta_column, num_model, data_folder, output_folder):
    pred_FVC = np.zeros((len(df),))
    pred_confidence = np.zeros((len(df),))

    model = build_model(len(meta_column))

    print('=' * 10, 'prediction', '=' * 10)
    
    for i in range(num_model):
        model.load_weights(output_folder + f'/weight_fold{i}.h5')
        
        pred = model.predict(x = get_item(df, meta_column, data_folder), batch_size = 32, verbose = 1)
            
        pred_FVC += pred[:,1]
        conf = pred[:,2] - pred[:,0]
        pred_confidence += conf
    
    pred_FVC /= num_model
    pred_confidence /= num_model
    
    return pred_FVC, pred_confidence