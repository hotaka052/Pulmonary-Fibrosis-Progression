import lightgbm as lgb
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from IPython.display import display

def train_model(x, y, lgbm_params, phase, output_folder):
    kf = KFold(n_splits = 5, random_state = 71, shuffle = True)
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(x)):
        print('=' * 10, 'Fold', fold, '=' * 10)
        
        X_train = x.iloc[tr_idx]
        y_train = y.iloc[tr_idx]
        
        X_val = x.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference = lgb_train)
        
        model = lgb.train(
            lgbm_params, 
            lgb_train, 
            valid_sets = [lgb_train, lgb_val],
            num_boost_round = 10000, 
            early_stopping_rounds = 100,
            verbose_eval = 1000,
        )
        
        train_pred = model.predict(X_train, num_iteration = model.best_iteration)
        val_pred = model.predict(X_val, num_iteration = model.best_iteration)

        """
        display(y_train.head())
        print(type(y_train))

        display(train_pred[:5])
        print(type(train_pred))
        """
        if phase == "fvc":
            train_loss = mean_absolute_error(y_train, train_pred)
            val_loss = mean_absolute_error(y_val, val_pred)
            
            print('Train Loss：{:.4f} | Val Loss：{:.4f}'.format(train_loss, val_loss))
            #lgb.plot_importance(fvc_model)
        
        with open(output_folder + f'/{phase}model-{fold}.pickle', mode = 'wb') as fp:
            pickle.dump(model, fp)
            