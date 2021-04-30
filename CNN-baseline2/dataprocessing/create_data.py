import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, scale

from .preprocessing import *

def get_y(df):
    df['y'] = np.nan
    
    for p in df.Patient.unique():
        sub = df.loc[df.Patient == p,:]
        fvc = sub.FVC.values
        weeks = sub.Weeks.values
        
        #weeksを縦の行列へ
        c = np.vstack([weeks, np.ones(len(weeks))]).T
        
        #weeksとfvcを最小二乗法で線形近似
        #a：傾き b：切片
        a, b = np.linalg.lstsq(c, fvc, rcond = -1)[0]
        
        df.loc[df.Patient == p, 'y'] = a
        
    return df

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

def base_fvc(train, test):
    """
    base_fvcカラムの設定
    """
    train_fvc = pd.DataFrame(train.groupby('Patient')['FVC'].mean())
    train_fvc.rename(columns = {'FVC' : 'base_fvc'}, inplace = True)
    train = pd.merge(train, train_fvc, on = 'Patient', how = 'left')
    train.drop('FVC', axis = 1, inplace = True)

    test['base_fvc'] = test['FVC']
    
    return train, test

def base_percent(train, test):
    """
    base_percentカラムの設定
    """
    train_percent = pd.DataFrame(train.groupby('Patient')['Percent'].mean())
    train_percent.rename(columns = {'Percent' : 'base_percent'}, inplace = True)
    train = pd.merge(train, train_percent, on = 'Patient', how = 'left')
    train.drop('Percent', inplace = True, axis = 1)
    train.drop_duplicates(keep = 'first', inplace = True, subset = 'Patient')
    train.reset_index(drop = True, inplace = True)

    test['base_percent'] = test['Percent']
    
    return train, test

def scaling(train, test):
    """
    Age,Weeksの正規化
    """
    train['Age'] /= 100
    train['Weeks'] /= 133
    test['Age'] /= 100
    test['Weeks'] /= 133

    scaler = StandardScaler()
    scaler.fit(train[['base_fvc', 'base_percent']])
    train[['base_fvc', 'base_percent']] = scaler.transform(train[['base_fvc', 'base_percent']])
    test[['base_fvc', 'base_percent']] = scaler.transform(test[['base_fvc', 'base_percent']])

    return train, test

def drop_data(train, test):
    """
    必要のないカラムの削除
    """
    drop_list = ['Sex', 'SmokingStatus']
    train.drop(drop_list, inplace = True, axis = 1)
    test.drop(drop_list, inplace = True, axis = 1)

    return train, test

def pfp_dataset(data_folder):
    """
    データの作成
    """
    train, test = read_data(data_folder)
    train = get_y(train)
    train, test = one_hot(train, test)
    train, test = base_fvc(train, test)
    train, test = base_percent(train, test)
    train, test = scaling(train, test)
    train, test = drop_data(train, test)

    return train, test