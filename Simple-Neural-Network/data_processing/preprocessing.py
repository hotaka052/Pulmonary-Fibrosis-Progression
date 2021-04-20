import pandas as pd

def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')
    test = pd.read_csv(data_folder + '/test.csv')

    #被っているデータの削除
    train.drop_duplicates(keep = False, inplace = True, subset = ['Patient', 'Weeks'])

    return train, test

def pre_sub(data_folder):
    sub = pd.read_csv(data_folder + '/sample_submission.csv')
    sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
    sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

    return sub