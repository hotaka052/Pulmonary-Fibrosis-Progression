import pandas as pd
from sklearn.preprocessing import StandardScaler, scale

from .preprocessing import *

def one_hot(train, test):
    """
    one_hot_encoder
    """
    df = pd.concat([train, test])
    sex_dummy = pd.get_dummies(df['Sex'], prefix = 'Sex')
    smoking_dummy = pd.get_dummies(df['SmokingStatus'], prefix = 'SS')
    df = pd.concat([df, sex_dummy], axis = 1)
    df = pd.concat([df, smoking_dummy], axis = 1)
    train = df.iloc[:len(train),:]
    test = df.iloc[len(train):,:]

    return train, test

def base_fvc(train, test, sub):
    """
    base_fvcカラムの作成
    """
    train['min_week'] = train['Weeks']
    train['min_week'] = train.groupby('Patient')['min_week'].transform('min')

    base = train.loc[train.Weeks == train.min_week]
    base = base[['Patient', 'FVC']]
    base.columns = ['Patient', 'base_FVC']

    train = pd.merge(train, base, on = 'Patient', how = 'left')
    train['base_week'] = train.Weeks - train.min_week

    test.rename(columns = {'Weeks' : 'min_week'}, inplace = True)

    test['base_FVC'] = test['FVC']
    test.drop('FVC', axis = 1, inplace = True)

    sub = pd.merge(sub, test, on = 'Patient', how = 'left')

    sub['base_week'] = sub.Weeks - sub.min_week

    return train, sub

def scaling(train, sub):
    """
    Age,Weeksの正規化
    """
    train['Age'] /= 100
    train['Weeks'] /= 133
    sub['Age'] /= 100
    sub['Weeks'] /= 133

    scaler = StandardScaler()
    scaler.fit(train[['Percent', 'base_FVC', 'base_week']])
    train[['Percent', 'base_FVC', 'base_week']] = scaler.transform(train[['Percent', 'base_FVC', 'base_week']])
    sub[['Percent', 'base_FVC', 'base_week']] = scaler.transform(sub[['Percent', 'base_FVC', 'base_week']])

    return train, sub

def drop_data(train, sub):
    """
    必要のないカラムの削除
    """
    drop_list = ['Sex', 'SmokingStatus', 'min_week']
    train.drop(drop_list, inplace = True, axis = 1)
    sub.drop(drop_list, inplace = True, axis = 1)

    return train, sub

def pfp_dataset(data_folder):
    """
    データの作成
    """
    train, test = read_data(data_folder)
    train, test = one_hot(train, test)
    sub = pre_sub(data_folder)
    train, sub = base_fvc(train, test, sub)
    train, sub = scaling(train, sub)
    train, sub = drop_data(train, sub)

    return train, sub