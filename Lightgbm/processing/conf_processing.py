import numpy as np
import pandas as pd
import scipy
import math
from functools import partial

def add_column(train, fvc_pred):
    conf_x = pd.concat([train, fvc_pred], axis = 1)
    conf_x.rename({0 : 'fvc_pred'}, axis = 1, inplace = True)
    difference = abs(train['FVC'] - conf_x.fvc_pred)
    conf_x['real_difference'] = difference
    conf_x['difference'] = abs(conf_x.fvc_pred - conf_x.base_FVC)
    conf_x['ratio'] = (conf_x.fvc_pred / conf_x.base_FVC)

    return conf_x

def loss_func(weight, row):
    confidence = weight
    delta = row['real_difference']
    score = math.sqrt(2) * delta / confidence + np.log(math.sqrt(2) * confidence)

    return score

def make_conf(df):
    results = []
    for i in range(len(df)):
        row = df.iloc[i]
        loss_partial = partial(loss_func, row = row)
        weight = [100]
        result = scipy.optimize.minimize(loss_partial, weight, method='SLSQP')
        x = result['x']
        results.append(x[0])
        
    return results

def drop_columns(df):
    drop_list = ['real_difference', 'FVC', 'Patient']
    df.drop(drop_list, axis = 1, inplace = True)

    return df

def conf_dataset(train, fvc_pred):
    conf_x = add_column(train, fvc_pred)
    conf_y = make_conf(conf_x)
    conf_x = drop_columns(conf_x)
    conf_y = pd.Series(conf_y)

    return conf_x, conf_y

def test_conf_dataset(test, inf_fvc_pred):
    inf_conf_x = pd.concat([test, inf_fvc_pred], axis = 1)
    inf_conf_x.rename({0 : 'fvc_pred'}, axis = 1, inplace = True)

    inf_conf_x['difference'] = abs(inf_conf_x.fvc_pred - inf_conf_x.base_FVC)
    inf_conf_x['ratio'] = (inf_conf_x.fvc_pred / inf_conf_x.base_FVC)

    return inf_conf_x