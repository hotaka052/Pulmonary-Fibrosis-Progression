import pandas as pd
#from IPython.display import display

from .lung_size import add_lung_size
from .preprocessing import *

def label_encoder(train, test):
    train.replace({'Sex':{'Male' : 0, 'Female' : 1}}, inplace = True)
    train.replace({'SmokingStatus':{'Currently smokes' : 0, 'Ex-smoker' : 1, 'Never smoked' : 2}}, inplace = True)
    test.replace({'Sex':{'Male' : 0, 'Female' : 1}}, inplace = True)
    test.replace({'SmokingStatus':{'Currently smokes' : 0, 'Ex-smoker' : 1, 'Never smoked' : 2}}, inplace = True)

    return train, test

def train_processing(train):
    """
    base_fvcとweek_passedの設定
    """
    train['min_week'] = train['predict_Week']
    train['min_week'] = train.groupby('Patient')['min_week'].transform('min')

    base = train.loc[train.predict_Week == train.min_week]
    base = base[['Patient', 'FVC']]
    base.columns = ['Patient', 'base_FVC']

    train = pd.merge(train, base, on = 'Patient', how = 'left')
    train['Week_passed'] = train.predict_Week - train.min_week

    train.drop('min_week', axis = 1, inplace = True)

    return train

def test_processing(test, sub):
    """
    testとsubの結合
    """
    rename_cols = {'Weeks' : 'base_Week', 'FVC' : 'base_FVC'}
    test.rename(columns = rename_cols, inplace = True)

    test = pd.merge(sub, test, on = 'Patient', how = 'left')

    test['Week_passed'] = test['predict_Week'] - test['base_Week']
    test.drop('base_Week', axis = 1, inplace = True)

    test.rename({'LungVolume_x' : 'LungVolume'}, inplace = True)

    return test

def fvc_dataset(data_folder):
    train, test = read_data(data_folder)
    sub = pre_sub(data_folder)

    train = add_lung_size(train, phase = 'train', data_folder = data_folder)
    test = add_lung_size(test, phase = 'test', data_folder = data_folder)

    train, test = label_encoder(train, test)
    train = train_processing(train)
    test = test_processing(test, sub)

    #display(train.head())
    #display(test.head())

    return train, test