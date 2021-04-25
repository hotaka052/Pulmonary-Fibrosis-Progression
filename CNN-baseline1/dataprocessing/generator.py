import numpy as np

from tensorflow.keras.utils import Sequence

from .processing_generator import *

class PFPGenerator(Sequence):
    """
    学習のためのgeneratorクラス
    返り値はx：画像、meta：テーブルデータ、y：目的変数
    """

    def __init__(self, df, meta, batch_size, data_folder):
        self.df = df
        self.meta = meta
        self.batch_size = batch_size
        self.data_folder = data_folder
        
    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        x , meta, y = [], [], []
        index = np.random.choice(len(self.df), size = self.batch_size) 
            
        for i in index:
            row = full_path(self.df.iloc[i], phase = 'train', data_folder = self.data_folder)
            img = get_img(row['full_path']) 
            img = mono_to_color(img) 
            img = img / 255.0
            x.append(img)
                
            y.append(self.df.iat[i,2])
            meta.append(self.df.iloc[i][self.meta])
            
        x = np.array(x, dtype = np.float32)
        meta = np.array(meta, dtype = np.float32)
        y = np.array(y, dtype = np.float32)
            
        return [x, meta], y

    def on_epoch_end(self):
        pass