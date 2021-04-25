import os
import numpy as np
import pydicom
import cv2

def full_path(df, phase, data_folder):
    """
    画像へのパスを設定する関数
    選ぶ画像はランダム
    最初と最後の方は肺が映っていなのでそこが選ばれた場合は選びなおし
    """

    p = df.Patient
    path_list = os.listdir(data_folder + f'/{phase}/{p}/')
    
    while True:
        img_path =  np.random.choice(path_list, size = 1)[0]
        if int(img_path[:-4]) / len(path_list) < 0.8 and int(img_path[:-4]) / len(path_list) > 0.2:
            df['full_path'] = (data_folder + f'/{phase}/{p}/{img_path}')
            break
        else:
            continue
        
    return df

def get_img(path):
    """
    pathを引数に画像を読み込む関数
    返り値はリサイズ済みの画像
    """
    d = pydicom.dcmread(path)
    img_resize = cv2.resize((d.pixel_array - d.RescaleIntercept) / (d.RescaleSlope * 1000), (256, 256))
    return img_resize

def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    1次元の画像を3次元に変換する関数
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data

    """
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V