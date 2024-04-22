"""
The main code page to run these models
"""
from Model_File.preprocessing import data_trans, read_file, split_data
from Model_File.models import LGBM, MCMC
from Model_File.plotting import *
from Model_File.packages import *


#Given which model 
model_type = sys.argv[-1]

#---------------------------- Model 1 ----------------------------#
if model_type == '-L':
    print('>>>>> Running the Light Gradient Boosting Model <<<<<')
    # read input from system terminal
    raw, file_name, config, mode = read_file.result
    # apply relative carry-over effect 
    new_data_path, new_data = data_trans.do(raw, file_name, config, mode)
    new_data.to_csv(new_data_path, index=0)
    # splitting new dataset into x and y
    x, y = split_data.do(new_data, config)
    # do the modelling part
    file_name_mode = f'{file_name}【{mode}】' #add mode into the filename
    model, folder_path = LGBM.do(x, y, config, file_name_mode)  
    # calculate elasticity and plot all response curves
    plot_allRC(x, y, raw, model, folder_path)
    print('\n>>>>> The Light Gradient Boosting Model is DONE !!! <<<<<\n')   
    
#---------------------------- Model 2 ----------------------------#       
if model_type == '-M':
    print('>>>>> Running the Markov chain Monte Carlo Model <<<<<')
    # 导入相应数据
    raw, file_name, feature, mode = read_file.result
    config = read_file.modelfig
    # Model Configures 
    n_points, model_name, stan_name, data_size, target_path = config.ParameterValue.values
    n_points = int(n_points)
    data_size = int(data_size)
    # Apply carry-over effect
    path, new_input = data_trans.do(raw, file_name, feature, mode)
    new_input.to_csv(path, index=0) #save the new dataset
    # Using Stanfile
    StanGenerator.generate(feature, stan_name)
    file_name_mode = f'{file_name}【{mode}】'
    # Modelling
    fit, folder_path = MCMC.do(file_name_mode, 
                                 new_input, 
                                 model_name, 
                                 stan_name, 
                                 feature, 
                                 data_size
                                )
    # Scatter plotting
    MCMC.Contribution(raw, folder_path, feature, fit, data_size)
    # Response curves plotting
    plotting_allRC(n_points, raw, feature, mode, fit, MCMC, folder_path)
    print('\n>>>>> The Markov chain Monte Carlo is DONE !!! <<<<<\n')
