import pydicom
import numpy as np
import pandas as pd
import os
import random
from scipy import ndimage
import shutil

import argparse

parser = argparse.ArgumentParser(
    description = ""
)

parser.add_argument('--data_folder', default = '/kaggle/input/osic-pulmonary-fibrosis-progression', type = str,
                    help = 'データの入っているフォルダ')

parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "アウトプットファイルを出力するフォルダ")

args = parser.parse_args()

def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')

    drop_list1 = train[train['Patient'] == 'ID00011637202177653955184'].index
    drop_list2 = train[train['Patient'] == 'ID00052637202186188008618'].index
    train.drop(drop_list1, inplace = True)
    train.drop(drop_list2, inplace = True)

    train.drop_duplicates(subset = 'Patient', inplace = True)
    train.reset_index(drop = True, inplace = True)

    return train

def get_3d_image(df, new_shape, dir_path, save_path):
    """
    CTスキャンを3D-imageに
    """
    p = df['Patient']
    path_list = os.listdir(dir_path + '/' + p)
    path_num =[]
    dataset = pydicom.dcmread(dir_path + '/' + p + '/' + path_list[0])
    d_array = np.zeros((dataset.Rows, dataset.Columns, len(path_list)), dtype = dataset.pixel_array.dtype)
    
    for path in path_list:
        path_num.append(int(path.split('.')[0]))
    path_num.sort()
    
    for i, path in enumerate(path_num):
        dataset = pydicom.dcmread(dir_path + '/' + p + '/' + str(path) + '.dcm')
        d_array[:,:,i] = dataset.pixel_array
        
    d_array = d_array.transpose(2, 1, 0)
    #3D-imageのresample
    ds_factor = [float(w) / float(f) for w, f in zip(new_shape, d_array.shape)]
    image = ndimage.zoom(d_array, ds_factor)
    image = np.expand_dims(image, 0)
    
    image = image / 255.0
    
    fname = save_path + '/' + f'{p}'
    np.save(fname, image)

if __name__ == '__main__':
    save_path = args.output_folder + '/3d_image'
    os.mkdir(save_path)

    train = read_data(args.data_folder)

    for i in range(len(train)):
        get_3d_image(
            df = train.iloc[i],
            new_shape = (40, 128, 128),
            dir_path = args.data_folder + '/train',
            save_path = save_path
        )

    shutil.make_archive(args.output_folder + '/3d_image', 'zip', save_path)
    shutil.rmtree(save_path)