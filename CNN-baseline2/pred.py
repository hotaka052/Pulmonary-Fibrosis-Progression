import numpy as np

from build_model import build_model

from dataprocessing.processing_generator import *

def predict(df, meta_columns, num_model, q, data_folder, output_folder):
    df['a'] = 0.0
    df['b'] = 0.0

    model = build_model(len(meta_columns))
    
    for n in range(num_model):
    
        for i in range(len(df)):
            model.load_weights(output_folder + f'/weight_fold{i}.h5')
            x, meta = [], []
            row = full_path(df.iloc[i], phase = 'test', data_folder = data_folder)
            img = get_img(row['full_path'])
            img = (img - img.min()) / (img.max() - img.min())
            img = np.expand_dims(img, axis = -1)
            x.append(img)
        
            meta.append(df.iloc[i][meta_columns])
        
            x = np.array(x, dtype = np.float32)
            meta = np.array(meta, dtype = np.float32)
        
            _a = model.predict([x, meta])
            a = np.quantile(_a, q)
        
            b = df.at[df.index[i], 'FVC'] - a * df.at[df.index[i], 'Weeks']
        
            df.at[df.index[i], 'a'] += a
            df.at[df.index[i], 'b'] += b
            
    df['a'] /= num_model
    df['b'] /= num_model
    
    return df
