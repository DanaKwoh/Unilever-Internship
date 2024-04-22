"""
· This code page is another testing LGBM draft. It's pretty similar with the 'LGBM.py'.
· To use this test model, change code of modelling part on the "main.py" directly. Using 'LGBM_test1.do(x, y, config, file_name_mode)' instead and remember to import this function at ahead..
"""

from Model_File.packages import *
from Model_File.plotting import *
from Model_File.functions import *
from Model_File.preprocessing import data_trans
warnings.filterwarnings('ignore')


def do(X, y, config, file_name):
    rand_seed = 731

    # Train/Test set split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle = False)

    # 文件格式满足：第一行为header("Feature"与"Constraint"),之后为每个feature对应的单调性限制
    mc_ls = config.loc[config.Target == 1, 'Constraints'].values
        # ================================================================================ #
    
    
            #=========================初步调参=========================#
    params = {
        'learning_rate': [0.1, 0.09, 0.08, 0.07, 0.05],
        'max_depth': np.arange(3, 9, 1),
        'num_leaves': range(1, 257, 8),
        'n_estimators':range(10, 100, 10), 
        'max_bin':np.arange(50, 301, 10), 
        'min_data_in_leaf':np.arange(2, 21, 2), 
        'bagging_freq':[0], 
        'bagging_fraction':[1], 
        'min_sum_hessian_in_leaf':[1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    }
    
    # Initialize the model with pre-defined monotone constrains list
    my_lgb = lgb.LGBMRegressor(
        random_state=rand_seed, objective = 'regression', metric = 'rmse', mc = mc_ls
    )
    
    # Initialize parameter tuning object
    my_rs = RandomizedSearchCV(my_lgb, params, random_state=rand_seed, n_jobs = 4, cv = 5, scoring = 'neg_mean_squared_error')
    
    print("**********Training Started**********")
          
    my_lgb_rs = my_rs.fit(X_train, y_train,
                     eval_set=[(X_train, y_train),(X_test, y_test)],
                     eval_metric=['rmse','mae'],
                     eval_names=['Train','Test'],
                     early_stopping_rounds=15,
                     verbose=10,
                     categorical_feature='auto')
    
    # 得到最优参数
    num_leaves = my_lgb_rs.best_params_['num_leaves']
    n_estimators = my_lgb_rs.best_params_['n_estimators']
    min_sum_hessian_in_leaf = my_lgb_rs.best_params_['min_sum_hessian_in_leaf']
    min_data_in_leaf = my_lgb_rs.best_params_['min_data_in_leaf']
    max_depth = my_lgb_rs.best_params_['max_depth']
    max_bin = my_lgb_rs.best_params_['max_bin']
    learning_rate = my_lgb_rs.best_params_['learning_rate']
    bagging_freq = my_lgb_rs.best_params_['bagging_freq']
    bagging_fraction = my_lgb_rs.best_params_['bagging_fraction']
    
    if 2**max_depth < num_leaves:
        num_leaves = 2**max_depth - 1
        # ================================================================================ #
    
    
            #=========================加入正则化调参=========================#
    # 加入正则化，避免过拟合
    # 其余参数用初步调参得出的最优参数
    params = {
        'num_leaves': [num_leaves], # 要满足2**max_depth > num_leaves
        'n_estimators': [n_estimators], 
        'min_sum_hessian_in_leaf': [min_sum_hessian_in_leaf], 
        'min_data_in_leaf': [min_data_in_leaf], 
        'max_depth': [max_depth], 
        'max_bin': [max_bin], 
        'learning_rate': [learning_rate], 
        'bagging_freq': [bagging_freq], 
        'bagging_fraction': [bagging_fraction],
        
        # 正则化参数
        'lambda_l2': np.arange(0.0, 1.1, 0.1)
    }
    
    # Initialize the model with pre-defined monotone constrains list
    my_lgb = lgb.LGBMRegressor(
        random_state=rand_seed, objective = 'regression', metric = 'rmse', mc = mc_ls
    )
    
    # Initialize parameter tuning object
    my_rs = RandomizedSearchCV(my_lgb, params, random_state=rand_seed, n_jobs = 4, cv = 5, scoring = 'neg_mean_squared_error')
    
    my_lgb_rs = my_rs.fit(X_train, y_train,
                     eval_set=[(X_train, y_train),(X_test, y_test)],
                     eval_metric = ['rmse','mae'],
                     eval_names=['Train','Test'],
                     early_stopping_rounds=15,
                     verbose=10,
                     categorical_feature='auto')
        # ================================================================================ #
    
    
            #=========================在原始数据上训练模型=========================#
    opt_params = my_lgb_rs.best_params_
    
    final = lgb.LGBMRegressor(
        random_state=rand_seed, objective = 'regression', metric = 'rmse', mc = mc_ls, 
        **opt_params
    )

    final.fit(X, y, 
            eval_set=[(X, y)],
            eval_names=['complete'],
            eval_metric = ['rmse', 'mae'],
            early_stopping_rounds=15,
            verbose=10,
            categorical_feature='auto')
    print("\n**********Training Ended**********\n")

    print('>>>>>After turnning:\n', opt_params,'\n')
    #====================================================================================#




    #=========================模型输出=========================#    
    # 保存模型
    model_name = 'LGBM'
    model_path = "Output/Output_PickleModels/" + model_name + ".pkl"
    joblib.dump(final, model_path)
    print("**********Model Saved**********\n")

    #=========================模型表现=========================# 
    # 模型R方
    y_series = y.squeeze()
    r_squared = 1-(((final.predict(X) - y_series)**2).sum()/((y_series-y_series.mean())**2).sum())
    print(">>>>>Model R-squared: " + str(r_squared))
    print('\n>>>>>Feature Importances:\n', final.feature_importances_)
    
    # plotting the fittness
    folder_path = f'{file_name}'
    folder_path = plot_fittness(final, y, X, y_test, X_test,r_squared, folder_path)
    return final, folder_path



