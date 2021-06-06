lgbm_fvc_params = {
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric' : 'rmse',
    'learning_rate':0.01,
    'max_depth' : -1,
    'min_child_weight': 15,
    'reg_alpha':0.0,
    'subsample':0.8,
    'random_state': 42,
    'verbose': -1,
}

lgbm_conf_params = {
    'boosting_type':'gbdt',
    'objective':'regression',
    'metric' : 'rmse',
    'learning_rate':0.1,
    'max_depth': -1,
    'min_child_weight':1,
    'reg_alpha':0.0,
    'subsample':0.8,
    'random_state':71,
    'verbose': -1,
}
