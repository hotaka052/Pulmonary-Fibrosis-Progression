import pandas as pd

def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')
    test = pd.read_csv(data_folder + '/test.csv')

    #画像データが壊れいるので削除
    drop_list1 = train[train['Patient'] == 'ID00011637202177653955184'].index
    drop_list2 = train[train['Patient'] == 'ID00052637202186188008618'].index
    train.drop(drop_list1, inplace = True)
    train.drop(drop_list2, inplace = True)

    #被っているデータの削除
    train.drop_duplicates(keep = False, inplace = True, subset = ['Patient', 'Weeks'])
    
    train.rename(columns = {'Weeks' : 'predict_Week'}, inplace = True)


    return train, test

def pre_sub(data_folder):
    sub = pd.read_csv(data_folder + '/sample_submission.csv')
    sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
    sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
    sub = sub[['Patient', 'Weeks']]
    sub.rename(columns = {'Weeks' : 'predict_Week'}, inplace = True)

    return sub