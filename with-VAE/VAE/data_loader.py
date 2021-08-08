import numpy as np
import os
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader

class CTDataset(Dataset):
    def __init__(self, root_dir, file_list):
        self.root_dir = root_dir
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        image = np.load(self.root_dir + '/' + self.file_list[index])
            
        return image

def vae_dataloader(data_folder):
    file_list = os.listdir(data_folder)

    tr_file, val_file = train_test_split(file_list, test_size = 0.2, random_state = 71)

    tr_ds = CTDataset(root_dir = data_folder, file_list = tr_file)
    val_ds = CTDataset(root_dir = data_folder, file_list = val_file)
    
    batch_size = 8
    tr_dl = DataLoader(dataset = tr_ds, batch_size = batch_size, shuffle = True)
    val_dl = DataLoader(dataset = val_ds, batch_size = batch_size, shuffle = False)

    return tr_dl, val_dl