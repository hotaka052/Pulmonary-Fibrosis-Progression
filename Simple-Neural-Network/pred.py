import numpy as np

from build_model import build_model

def predict(df, meta, num_model, output_folder):
    pred_FVC = np.zeros((len(df),))
    pred_confidence = np.zeros((len(df),))

    model = build_model(len(meta))
    
    for i in range(num_model):
        model.load_weights(output_folder + f'/weight_fold{i}.h5')
        
        pred = model.predict(x = df[meta], batch_size = 64, verbose = 1)
            
        pred_FVC += pred[:,1]
        conf = pred[:,2] - pred[:,0]
        pred_confidence += conf
    
    pred_FVC /= num_model
    pred_confidence /= num_model
    
    return pred_FVC, pred_confidence