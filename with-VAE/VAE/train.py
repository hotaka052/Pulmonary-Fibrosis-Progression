import numpy as np
import time
import warnings

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from seed_everything import seed_everything
from model import VAE
from data_loader import vae_dataloader

import argparse

parser = argparse.ArgumentParser(
    description = ""
)

parser.add_argument('--data_folder', type = str,
                    help = 'データの入っているフォルダ')

parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "アウトプットファイルを出力するフォルダ")

args = parser.parse_args()

def train_model(epochs, es_patience, tr_dl, val_dl):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = VAE(latent_dim = 20, image_shape = (1, 40, 128, 128))
    
    model_path = args.output_folder + f'/model.pth'
    
    patience = es_patience
    best_loss = np.inf
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, 
        mode = 'min', 
        patience = 3, 
        verbose = True, 
        factor = 0.2
    )
    
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        val_loss = 0
        
        model.train()
        
        for x in tr_dl:
            x = torch.tensor(x, device = device, dtype = torch.float32)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        model.eval()
        
        with torch.no_grad():
            for x_val in val_dl:
                x_val = torch.tensor(x_val, device = device, dtype = torch.float32)
                z_val = model(x_val)
                loss = criterion(z_val, x_val)
                
                val_loss += loss.item()
            finish_time = time.time()
                
            print("Epoch：{:03} | Train Loss：{:.4f} | Val Loss：{:.4f} | Learning Time：{:.4f}"
                  .format(epoch + 1, train_loss, val_loss, finish_time - start_time))
                
            scheduler.step(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience = es_patience
                
                torch.save(model, model_path)
                
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best Val Loss：{:.3f}'.format(best_loss))
                    break

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    seed_everything(71)

    tr_dl, val_dl = vae_dataloader(args.data_folder)
    
    train_model(
        epochs = 300,
        es_patience = 10,
        tr_dl = tr_dl,
        val_dl = val_dl,
    )