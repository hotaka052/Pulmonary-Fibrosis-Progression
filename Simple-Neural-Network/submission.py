import pandas as pd

from data_processing import pfp_dataset
from train import train_model
from pred import predict

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
    meta.remove('FVC')

    return meta

def submission(pred_FVC, pred_confidence):
    sub = pd.read_csv(args.data_folder + '/sample_submission.csv')
    sub['FVC'] = pred_FVC
    sub['Confidence'] = pred_confidence
    sub.to_csv(args.output_folder + '/submission.csv', index = False)

if __name__ == '__main__':
    train, sub = pfp_dataset(args.data_folder) # データの準備
    train_model(
        df = train,
        meta = select_meta(train),
        epochs = args.epochs,
        output_folder = args.output_folder
    )

    pred_FVC, pred_confidence = predict(
        df = sub,
        meta = select_meta(train),
        num_model = 5,
        output_folder = args.output_folder
    )

    submission(pred_FVC, pred_confidence)
