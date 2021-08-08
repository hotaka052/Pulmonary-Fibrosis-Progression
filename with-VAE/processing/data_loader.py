import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

from .processing_dataloader import *

class PFPDataset(Dataset):
    def __init__(self, df, vae_model_path, image3d_folder, meta_features):
        self.df = df
        self.vae_model = torch.load(vae_model_path)
        self.image3d_folder = image3d_folder
        self.meta_features = meta_features

    def __getitem__(self, index):
        row = full_path(self.df.iloc[index], self.image3d_folder)
        x = get_latent_dim(row['full_path'], self.vae_model)
        x = torch.squeeze(x)
        
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype = np.float32)

        y = self.df.iloc[index]['FVC']
        
        return (x, meta), y

    def __len__(self):
        return len(self.df)

def pfp_dataloader(train, vae_model_path, image3d_folder, meta):
    train, val = train_test_split(train, test_size = 0.2, random_state = 71)
    tr_ds = PFPDataset(
        df = train, 
        vae_model_path = vae_model_path, 
        image3d_folder = image3d_folder,
        meta_features = meta
    )

    val_ds = PFPDataset(
        df = val, 
        vae_model_path = vae_model_path, 
        image3d_folder = image3d_folder,
        meta_features = meta
    )

    tr_dl = DataLoader(dataset = tr_ds, batch_size = 64, shuffle = True)
    val_dl = DataLoader(dataset = val_ds, batch_size = 16, shuffle = True)

    return tr_dl, val_dl