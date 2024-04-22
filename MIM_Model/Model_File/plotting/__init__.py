"""
This code page is used to plot. Generally are response curves and fittness.
    ·To use them, try 'from Model_File.plotting import *'.
    ·Then you can use any of these functions.
"""

from Model_File.packages import *
from Model_File.functions import *
from Model_File.preprocessing import data_trans


#========================= Plotting =========================#
"""LGBM"""
# plot the response curve of one given feature within the range [0.5,1.51]
def response_curve_half(feature, x, y, model):
    x_li, y_li = [], []

    for i in range(7,20):
        change = i*0.07
        x_new = x.copy()
        x_new[feature] *= change
        x_li.append((x[feature].mean())*change)
        y_pre = model.predict(x_new)
        y_li.append(y_pre.mean())
    return x_li, y_li #return corresponding x-axis and y-axis values


"""LGBM"""
# plot the response curve of one given feature within the range [min, max]
def response_curve_one(feature, x, y, raw, model):
    x_li, y_li = [], []
    raw = raw.sort_values(feature)
    f_data = raw[feature]
    for i in f_data.values:
        x_new = x.copy()
        point = i / f_data.mean()      
        x_new[feature] *= point
        x_li.append(i)
        y_pre = model.predict(x_new)
        y_li.append(y_pre.mean())
        
    return f_data.mean(), x_li, y_li #return corresponding x-axis and y-axis values


"""MCMC"""
# plot the response curve of one feature by given num of point
def response_curve_MCMC(f,n_points,data,feature,mode,fit,MCMC,folder_path): 
    x_li, y_li = [], []
    data_size = data.shape[0]
    mcoefs_raw, mcoefs_raw_intercept = MCMC.inverse_coef(data, 
                                                         data_size, 
                                                         feature, 
                                                         fit
                                                        )
    # calculating all values
    for i in range(-n_points, n_points+1):
        delta = i * 0.05
        x, predicted_y = MCMC.predict(f, 
                                      data, 
                                      folder_path, 
                                      feature, 
                                      mode,
                                      delta, 
                                      mcoefs_raw, 
                                      mcoefs_raw_intercept
                                     )
        x_li.append(x)
        y_li.append(predicted_y)
    
    return x_li, y_li #return corresponding x-axis and y-axis values


"""LGBM"""
# plotting the model's behavior and fittness file_name_mode
def plot_fittness(model, y, x, y_test, x_test, r_squared, file_name_mode):
    # Crate a new folder and add time stamp and suffix
    now = datetime.now().date()
    t = now.strftime("%m-%d")
    i = 0
    while(True):
        folder_path = f'Output/Output_Plots/L-{file_name_mode}{t}_{i}'
        try:
            os.mkdir(folder_path) 
        except FileExistsError:
            i += 1
            pass
        else:
            break
    folder_path = f'Output/Output_Plots/L-{file_name_mode}{t}_{i}'
    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(model.predict(x),label = 'Prediction')
    plt.plot(y,label='Ground Truth')
    plt.grid()
    plt.legend()
    plt.title(f'Actual vs Prediction -- R2: {r_squared.round(4)}')
    plt.savefig(f'{folder_path}/LGBM_Actual_VS_Predict.png')
    plt.close()
    print('**********Done Plotting Fittness **********')
    return folder_path


"""MCMC"""
# plotting the model's behavior and fittness file_name_mode
def plot_predict(file_name_mode, prediction, actual,  data_size, r2):
    timeIDX = list(range(1, data_size+1))
    # Crate a new folder and add time stamp and suffix
    now = datetime.now().date()
    t = now.strftime("%m-%d")
    i = 0
    while(True):
        folder_path = f'Output/Output_Plots/M-{file_name_mode}{t}_{i}'
        try:
            os.mkdir(folder_path) 
        except FileExistsError:
            i += 1
            pass
        else:
            break
    folder_path = f'Output/Output_Plots/M-{file_name_mode}{t}_{i}'
    # Plotting
    plt.figure(figsize=(10,5))
    plt.plot(timeIDX, prediction, label = "Prediction")
    plt.plot(timeIDX, actual, label = "Actual")
    plt.legend()
    plt.xticks(timeIDX)
    plt.xlabel('Time')
    plt.ylabel('Unit')
    plt.title(f'Actual vs Prediction -- R2: {r2.round(4)}')
    plt.savefig(f'{folder_path}/MCMC_Actual_VS_Predict.png')
    plt.close()
    print('**********Done Plotting Fittness **********')
    return folder_path


"""LGBM"""
# plotting one response curve
def plot_RC(elasticity, coordinates, feature, folder):
    plt.figure(figsize = (8,5))
    plt.plot(coordinates[1], coordinates[2])
    plt.axvline(coordinates[0], color = 'r')
    plt.axvline(coordinates[0]*0.95, color = 'g')
    plt.axvline(coordinates[0]*1.05, color = 'g')
    plt.title(f'{feature.name} has elasticity*5:   {elasticity}')
    plt.savefig(f'{folder}/{feature.name}.png')
    plt.close()

    
"""LGBM"""
# plotting all reponse curves
def plot_allRC(x, y, raw, model, folder_path):
    E_table = {'Feature':[], 'Importance':[], 'e5-L':[], 'e5-R':[], 'e5-avg':[]}
    folder = f'{folder_path}/Elastic_Curves'
    os.mkdir(folder)
    E_table['Importance'] = model.feature_importances_
    
    for feature in x.columns:
        E = elasticity(feature, x, y, model)
        E_table['Feature'].append(feature)
        E_table['e5-L'].append(f'{E[0]}%')
        E_table['e5-R'].append(f'{E[1]}%')
        E_table['e5-avg'].append(f'{np.mean(E).round(4)}%')
        coordinates = response_curve_one(feature, x, y, raw, model)
        plot_RC(E, coordinates, x[feature], folder)
    E_table = pd.DataFrame(E_table)
    E_table.to_csv(f'{folder_path}/Elasticity.csv', index = None)
    print('**********Done Plotting Response Curve **********')
    
    
"""MCMC""" 
# plotting all reponse curves
def plotting_allRC(n, data, feature, mode, fit, MCMC, folder_path):
    # 准备保存数据
    Elasticity = {'Feature':[], 'Left Elasticity':[], 'Right Elasticity':[]}
    
    # 准备保存图像
    os.makedirs(f"{folder_path}/Response_Curves") 
    features = feature.copy()
    features_list = features.Feature.values[1:]
    
    for f in features_list:
        x_li, y_li = response_curve_MCMC(f,
                                         n,
                                         data,
                                         feature,
                                         mode,
                                         fit,
                                         MCMC,
                                         folder_path
                                        )
        
        left_elasticity = round(5 * ((y_li[n] - y_li[n-1])/ y_li[n-1])/0.05, 4)
        right_elasticity = round(5 * ((y_li[n+1] - y_li[n])/ y_li[n])/0.05, 4)
        
        # 记录elasticity数据
        Elasticity['Feature'].append(f)
        Elasticity['Left Elasticity'].append(left_elasticity)
        Elasticity['Right Elasticity'].append(right_elasticity)
        
        # 画出Response Curve 曲线
        plt.plot(x_li, y_li)
        plt.title(f'{f} has elasticity : [{left_elasticity}, {right_elasticity}]')
        plt.axvline(x_li[n-1], color='green')
        plt.axvline(x_li[n+1], color='green')
        plt.axvline(x_li[n], color='red')
        plt.savefig(f"{folder_path}/Response_Curves/{f}.png")
        plt.close()
    
    # 保存elasticity数据
    Elasticity = pd.DataFrame(Elasticity)
    Elasticity['Avg Elasticity'] = round((Elasticity['Left Elasticity'] + Elasticity['Right Elasticity'])/2, 4)
    Elasticity.to_csv(f'{folder_path}/Elasticity.csv', index = 0)  
    print('**********Done Plotting Response Curve **********')


__all__ = [
    'response_curve_half',
    'response_curve_one',
    'response_curve_MCMC',
    'plot_fittness',
    'plot_predict',
    'plot_RC',
    'plot_allRC',
    'plotting_allRC'
    ]

