import numpy as np
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from losses import mloss, score

def train_model(model, train_dl, val_dl, epochs, es_patience):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用デバイス：', device)
    
    model.to(device)
    
    criterion = mloss(0.8, device = device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    
    scheduler = ReduceLROnPlateau(
        optimizer = optimizer, 
        mode = 'min', 
        patience = 3, 
        verbose = True, 
        factor = 0.2
    )
    
    patience = es_patience
    
    for epoch in range(epochs):
        start_time = time.time()
        
        train_loss = 0
        val_loss = 0
        train_score = 0
        val_score = 0
        best_score = np.inf
        
        model.train()
            
        for x, y in train_dl:
            x[0] = torch.tensor(x[0], device = device, dtype = torch.float32)
            x[1] = torch.tensor(x[1], device = device, dtype = torch.float32)
            y = torch.tensor(y, device = device, dtype = torch.float32)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(y, z)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_score += score(y, z, device)
                
        model.eval()
            
        with torch.no_grad():
            for x_val, y_val in val_dl:
                x_val[0] = torch.tensor(x_val[0], device = device, dtype = torch.float32)
                x_val[1] = torch.tensor(x_val[1], device = device, dtype = torch.float32)
                y_val = torch.tensor(y_val, device = device, dtype = torch.float32)
                z_val = model(x_val)
                loss = criterion(y_val, z_val)
                val_loss += loss.item()
                val_score += score(y_val, z_val, device)
                    
            finish_time = time.time()
                
            print("Epoch：{:03} | Train Loss：{:.4f} | Train Score：{:.4f} | Val Loss：{:.4f} | Val Score：{:.4f} | Training Time：{:.4f}".format(
            epoch + 1, train_loss, train_score, val_loss, val_score, finish_time - start_time))
                
            scheduler.step(val_loss)
                
            if val_score <= best_score:
                best_score = val_score
                patience = es_patience
                    
            else:
                patience -= 1
                if patience == 0:
                    print('Early Stopping. Best Score：{:.4f}'.format(best_score))
                    break

    return model