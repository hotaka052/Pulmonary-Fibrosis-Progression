import os
from tqdm.notebook import tqdm

from .lung_size_processing import ThrDetector, AreaIntegral

"""
肺の大きさをカラムに追加する
https://www.kaggle.com/khyeh0719/lung-volume-calculus-with-trapezoidal-rule
"""

BAD_ID = ['ID00011637202177653955184', 'ID00052637202186188008618']

def add_lung_size(df, phase, data_folder):
    integral = AreaIntegral(ThrDetector())

    df_data = {}

    for p in df.Patient.values:
        df_data[p] = os.listdir(data_folder + f'/{phase}/{p}/')
    
    keys = [k for k in list(df_data.keys()) if k not in BAD_ID]

    volume = {}
    for k in tqdm(keys, total = len(keys)):
        x = []
        for i in df_data[k]:
            x.append(data_folder + f'/{phase}/{k}/{i}') 
        volume[k] = integral(x)

    for k in tqdm(df.Patient.values):
        if k in BAD_ID:
            continue
        df.loc[df.Patient == k,'LungVolume'] = volume[k]

    return df