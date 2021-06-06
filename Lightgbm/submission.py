import pandas as pd
from IPython.display import display

from processing import *
from train import train_model
from config import *
from pred import predict

import warnings

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
    meta = list(train.columns)
    meta.remove('Patient')
    meta.remove('FVC')

    return meta

def make_submission(fvc_pred, conf_pred):
    sub = pd.read_csv(args.data_folder + '/sample_submission.csv')
    sub['FVC'] = fvc_pred
    sub['Confidence'] = conf_pred

    sub.to_csv(args.output_folder + '/submission.csv', index = False)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    train, test = fvc_dataset(args.data_folder)
    meta = select_meta(train)

    train_model(
        x = train[meta],
        y = train['FVC'], 
        lgbm_params = lgbm_fvc_params,
        phase = 'fvc',
        output_folder = args.output_folder,
    )

    fvc_pred = predict(train[meta], phase = 'fvc', output_folder = args.output_folder)

    #display(fvc_pred.head())

    conf_x, conf_y = conf_dataset(train, fvc_pred)
    train_model(
        x = conf_x,
        y = conf_y,
        lgbm_params = lgbm_conf_params,
        phase = 'conf',
        output_folder = args.output_folder,
    )

    inf_fvc_pred = predict(test[meta], phase = 'fvc', output_folder = args.output_folder)

    inf_conf_x = test_conf_dataset(test[meta], inf_fvc_pred)

    inf_conf_pred = predict(inf_conf_x, phase = 'conf', output_folder = args.output_folder)

    make_submission(inf_fvc_pred, inf_conf_pred)
