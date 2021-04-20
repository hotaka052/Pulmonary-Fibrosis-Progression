import numpy as np

from tensorflow.keras.utils import Sequence

class PFPGenerator(Sequence):
    def __init__(self, df, meta, batch_size):
        self.df = df
        self.meta = meta
        self.batch_size = batch_size
        
    def __len__(self):
        return self.batch_size
    
    def __getitem__(self, idx):
        x, y = [], []
        index = np.random.choice(len(self.df), size = self.batch_size) 
            
        for i in index:
            y.append(self.df.iat[i,2])
            x.append(self.df.iloc[i][self.meta])
              
        x = np.array(x, dtype = np.float32)
        y = np.array(y, dtype = np.float32)
            
        return x, y
    
    def on_epoch_end(self):
        pass