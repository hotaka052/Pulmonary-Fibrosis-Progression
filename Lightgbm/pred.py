import numpy as np
import pandas as pd
import pickle

def predict(x, phase, output_folder, num_model = 5):
    pred = np.zeros(len(x))
    x = x.values
    for i in range(num_model):
        with open(output_folder + f'/{phase}model-{i}.pickle', mode = 'rb') as fp:
            model = pickle.load(fp)
            
            pred += model.predict(x, num_iteration = model.best_iteration)
            
    pred /= num_model
    pred = pd.Series(pred)
            
    return pred