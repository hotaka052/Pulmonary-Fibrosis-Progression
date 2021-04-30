import os
import numpy as np
import pydicom
import cv2

def full_path(df, phase, data_folder):
    """
    受け取ったデータに'full_path'カラムを追加する関数
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
