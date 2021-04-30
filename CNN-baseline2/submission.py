import pandas as pd

from dataprocessing import pfp_dataset, pre_sub
from train import train_model
from pred import *

#====================
# logの削除
#====================
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import logging
tf.get_logger().setLevel(logging.ERROR)

import argparse

parser = argparse.ArgumentParser(
    description = "parameter for training"
)

parser.add_argument('--data_folder', default = '/kaggle/input/osic-pulmonary-fibrosis-progression', type = str,
                    help = 'データの入っているフォルダ')
parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "提出用ファイルを出力するフォルダ")
parser.add_argument('--epochs', default = 300, type = int,
                    help = "何エポック学習するか")

args = parser.parse_args()

def select_meta(df):
    meta = list(df.columns)
    meta.remove('Patient')
    meta.remove('Weeks')
    meta.remove('y')

    return meta

def submission(test, sub):
    for k in sub.Patient_Week.values:
        p, w = k.split('_')
        w = int(w) 
        row = test.loc[test.Patient == p]
    
        # y = ax + b
        fvc = row['a'].values[0] * w + row['b'].values[0]
        sub.loc[sub.Patient_Week == k, 'FVC'] = fvc
        sub.loc[sub.Patient_Week == k, 'Confidence'] = (
            row['Percent'].values[0] - row['a'].values[0] * abs(row['Weeks'].values[0] - w) 
        ) 
    
    sub.to_csv(args.output_folder + '/submission.csv', index = False)

if __name__ == '__main__':
    train, test = pfp_dataset(args.data_folder) # データの準備
    train_model(
        df = train,
        meta_column = select_meta(train),
        epochs = args.epochs,
        data_folder = args.data_folder,
        output_folder = args.output_folder
    )

    test = predict(
        df = test,
        meta_columns = select_meta(train),
        num_model = 5,
        q = 0.1,
        data_folder = args.data_folder,
        output_folder = args.output_folder
    )

    sub = pre_sub(args.data_folder)

    submission(test, sub)