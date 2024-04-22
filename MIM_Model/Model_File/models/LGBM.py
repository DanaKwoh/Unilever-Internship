"""
· This code page is LightGB Model. It contains only one function called "do". 
· To use this model on the 'main.py', please remember to import it at ahead... 'from Model_File.models import LGBM'

整体思路：
    1. Split trainning and testing.
    2. Initial parameters setting on LGBMRegressor
    3. Randomized Search Cross Validation
    4. Train a new LGBMRegessor with optimal parameters
    5. Display the tuning parameter.
    6. Create a new folder under "Output/Output_Plots": [folder_name] => 'model type'; 'file name'; 'carry-over method'; 'training date'; 'suffix'
    7. Plotting the fittness and R-Square of the final model, saved in the folder.
    8. Return the final model & path of the new folder
"""

from Model_File.packages import *
from Model_File.plotting import *
from Model_File.functions import *
from Model_File.preprocessing import data_trans
warnings.filterwarnings('ignore')


def do(X, y, config, file_name_mode):
    rand_seed = 731

    # Train&Test set split, no randomly selecting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle = False)

    # constraints list looks for: when target==1.
    mc_ls = config.loc[config.Target == 1, 'Constraints'].values
    
#========================= 在原始数据上训练模型 =========================#
    
    #需要根据特征量的不同进行适当地改变
    params = {
        'learning_rate': [0.1, 0.09, 0.08, 0.07, 0.05],
        'max_depth': range(6, 15, 1),
        'num_leaves': range(20, 100, 1),
        'n_estimators':range(100, 200, 5),   
        'max_bin':range(50, 350, 10), 
        'min_data_in_leaf':range(2, 21, 2), 
    }
        
    my_lgb = lgb.LGBMRegressor(
        random_state=rand_seed, 
        objective = 'regression', #options: regression_l1; huberl; possion; gammal
        metric = ['rmse','mae'], 
        mc = mc_ls, #constraints单调性
        importance_type='gain', #options = 'split'; 'gain'
        zero_as_missing = True,      
#         boosting = 'rf'
    )
    
    # Initialize parameter tuning object
    my_rs = RandomizedSearchCV(
        my_lgb, 
        params, 
        random_state = rand_seed, 
        cv = 3, #cross-validation generator or an iterable, num of fold.
        n_jobs = -1, # Num of jobs to run in parallel, while -1 means using all processors
        scoring = 'neg_mean_squared_error'
    )
    
    my_lgb_rs = my_rs.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train),(X_test, y_test)],
        eval_metric = ['rmse','mae'],
        eval_names=['Train','Test'],
        early_stopping_rounds=150,
        verbose=100
    )
    
    # Pick the optimal parameters from above
    opt_params = my_lgb_rs.best_params_
    
    # Train the final model
    final = lgb.LGBMRegressor(
        objective = 'regression', 
        metric = ['rmse','mae'], 
        mc = mc_ls, 
        importance_type='gain', # options = 'split'; 'gain'
        zero_as_missing = True, 
        random_state=rand_seed,
        **opt_params,
#         boosting = 'rf'
    )
    final.fit(
        X, y,
        eval_set=[(X, y)],
        eval_names=['complete'],
        eval_metric = ['rmse', 'mae'],
        early_stopping_rounds=30,
        verbose=10, 
    )
    
    print("\n********** Training Ended **********\n")
    print('>>>>> After tuning parameters:\n', opt_params,'\n')
    #==============================================================================#

    #=========================模型输出=========================#    
    # 保存模型pkl格式
    model_name = 'LGBM'
    model_path = "Output/Output_PickleModels/" + model_name + ".pkl"
    joblib.dump(final, model_path)
    print("********** Model Saved **********\n")

    #=========================模型表现=========================# 
    # 模型R方
    y_series = y.squeeze()
    r_squared = 1-(((final.predict(X) - y_series)**2).sum()/((y_series-y_series.mean())**2).sum())
    print(">>>>> Model R-squared: " + str(r_squared))
#     print('\n>>>>> Feature Importances:\n', final.feature_importances_)
    
    #========================= Plotting the Fittness =========================#
    folder_path = plot_fittness(final, y, X, y_test, X_test,r_squared, file_name_mode)
    
    
    return final, folder_path



