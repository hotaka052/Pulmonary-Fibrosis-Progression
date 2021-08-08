import numpy as np

import torch

def full_path(df, image3d_folder):
    """
    画像へのパスを設定する関数
    """
    p = df.Patient
    df['full_path'] = (image3d_folder + f'/{p}.npy')
    
    return df

def get_latent_dim(path, model):
    """
    pathを引数に画像を読み込む関数
    返り値はリサイズ済みの画像
    """
    image = np.load(path)
    image = np.expand_dims(image, 0)
    image = torch.tensor(image, device = 'cuda', dtype = torch.float32)
    
    model.eval()
    with torch.no_grad():
        x = model.encoder(image)
    
    return x