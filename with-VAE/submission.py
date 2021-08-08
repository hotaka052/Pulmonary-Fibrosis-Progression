from processing import *
from model import NN
from train import train_model
from VAE.seed_everything import seed_everything

import argparse

parser = argparse.ArgumentParser(
    description = "Parameter for training"
)
parser.add_argument('--data_folder', default = '/kaggle/input/osic-pulmonary-fibrosis-progression', type = str,
                    help = 'データの入っているフォルダ')
parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "提出用ファイルを出力するフォルダ")
parser.add_argument('--vae_model_path', type = str,
                    help = "VAEモデルへのパス")
parser.add_argument('--image3d_folder', type = str,
                    help = '3D imageへのパス')
parser.add_argument('--epochs', default = 300, type = int,
                    help = "何エポック学習するか")
parser.add_argument('--es_patience', default = 10, type = int,
                    help = 'どれだけ改善が無かったら学習を止めるか')

args = parser.parse_args()

def select_meta(df):
    """
    テーブルデータの中で使うカラムを設定
    """
    meta = list(df.columns)
    meta.remove('Patient')
    meta.remove('FVC')

    return meta

if __name__ == '__main__':
    seed_everything(71)

    train, sub = pfp_dataset(args.data_folder)

    meta = select_meta(train)

    tr_dl, val_dl = pfp_dataloader(
        train = train, 
        vae_model_path = args.vae_model_path, 
        image3d_folder = args.image3d_folder,
        meta = meta,
    )

    model = NN(n_meta = len(meta), latent_dim = 20)

    model = train_model(
        model = model,
        train_dl = tr_dl,
        val_dl = val_dl,
        epochs = args.epochs,
        es_patience = args.es_patience
    )