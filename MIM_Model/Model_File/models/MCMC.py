"""
· This code page is MCMC Model. It contains only four functions called 
    ['do'; 'Contribution'; 'inverse_coef'; 'predict']. 
· To use this model on the 'main.py', please remember to import it at ahead... 'from Model_File.models import MCMC'
· If you want to use them, for example try 'MCMC.do()'

"""

from Model_File.packages import *
from Model_File.preprocessing import data_trans
from Model_File.plotting import *



#========================= Modelling =========================#
def do(file_name_mode, data_processed, model_name, stan_name, feature, data_size):
    
    #============ Data Preparation ============#  
    features = feature.copy()
    
    # 确认数据集行数
    N = data_processed.shape[0]
    data_processed = data_processed.iloc[-data_size:,]
    if data_size > N:
        print("所指定时间范围超过原数据时间范围！！请指定小于等于" + str(N) + "的时间范围！！")
        sys.exit()
        
    # 取出需要用到的特征名列表
    features_list = features['Feature'].values
    data = data_processed[features_list]
    
    # 标准化数据
    data = (data-data.mean(axis=0))/data.std(axis=0)
    
    # 把数据转换成stan可以接收的形式
    feature_list_stan = [i.replace("&", "").replace("-", "_").replace(".", "_") for i in features_list]
    data_stan = {'N': data_size}
    for i, val in enumerate(feature_list_stan):
        data_stan[val] = data[features_list[i]]
        
    #============ Model Building ============#
    print("**********Building Started**********")
    model_file = "Output/Output_StanFiles/" + stan_name
    sm = pystan.StanModel(file=model_file)
    fit = sm.sampling(data=data_stan, control=dict(adapt_delta=0.95), n_jobs = 1)
    print(fit)
    
    # Model diagnostics
    stan_utility.check_all_diagnostics(fit)
    print("**********Building Ended**********")
    
    # 保存模型 
    model_name = "Output/Output_PickleModels/" + model_name + ".pkl"
    with open(model_name, 'wb') as f:
        pickle.dump(sm, f)
    print("**********Model Saved**********")
      
        
    #============ Visualization ============#
    prediction = fit.extract()['prediction'].mean(axis=0)
    actual = data_stan[feature_list_stan[0]].values    
    # 模型R方
    r_squared_norm = 1-(((prediction - actual)**2).sum()/((actual-actual.mean())**2).sum())    
    folder_path = plot_predict(file_name_mode, prediction, actual,  data_size, r_squared_norm)    
    print(">>>>>Model R-squared: " + str(r_squared_norm))      
    return fit, folder_path


#========================= Contribution =========================#
def Contribution(raw, path, feature, fit, data_size):
    features = feature.copy()
    features_list = list(features['Feature'].values)    
    param_list = fit.get_posterior_mean()
    coefs_list = []
    for coef in param_list:
        mean = coef.mean()
        coefs_list.append(mean)
    
    # 模型系数
    n_coefs = len(features_list) - 1
    coefs = coefs_list[0:n_coefs]
    intercept = coefs_list[n_coefs]
    
    # Specify the time range of the model
    raw = raw.iloc[-data_size:,]
    
    raw = raw[features_list]
    X = raw.iloc[:,1:]
    y = raw.iloc[:,0]
    
    # 去标准化后的模型系数
    mcoefs_raw = [coefs[col] * y.std() / X.iloc[:,col].std() for col in range(0,X.shape[1])]
    
    # 去标准化后的intercept
    mcoefs_raw_intercept = intercept * y.std() + y.mean() - X.dot(mcoefs_raw).mean()
    
    # 去标准化后的prediction
    y_raw_hat = X.dot(mcoefs_raw) + mcoefs_raw_intercept
    
    # 去标准化后的模型R方
    r_squared_denorm = 1-(((y_raw_hat - y)**2).sum()/((y-y.mean())**2).sum())
    
    # 得到并导出feature contribution的excel文件
    # 第一个sheet，保存原数据
    contribution = X * mcoefs_raw
    contribution['Intercept'] = mcoefs_raw_intercept
    contribution['Prediction'] = y_raw_hat
    contribution['R squared'] = r_squared_denorm
    
    # 第二个sheet，计算整年份的feature contribution，若不满一年，则计算包含所有数据
    N_year = data_size // 12
    data_sheet2 = contribution.iloc[:,:-1]
    
    feature_column = list(data_sheet2.columns)[:-1]
    feature_contribution = pd.DataFrame({"Feature": feature_column})
    
    if N_year == 0:
        sum_by_feature = data_sheet2.sum().to_frame().T
        sum_prediction = sum_by_feature["Prediction"].values[0]
        for j in list(sum_by_feature.columns):
            sum_by_feature[j] = sum_by_feature[j] / sum_prediction * 100.0
        feature_contribution["Contribution(%)_all"] = sum_by_feature.values[0][:-1]
        
    for i in range(N_year):
        sum_by_feature = data_sheet2.iloc[-12 * (i+1):,].sum().to_frame().T
        sum_prediction = sum_by_feature["Prediction"].values[0]
        for j in list(sum_by_feature.columns):
            sum_by_feature[j] = sum_by_feature[j] / sum_prediction * 100.0
        feature_contribution["Contribution(%)_" + "latest " + str(i+1) + " year"] = sum_by_feature.values[0][:-1]
        
    # 第三个sheet，基于Target Feature Config，计算其在原数据中占比与模型中占比，并包括其他作图所需数据
    # Load Target Features Config
    target_features = feature.loc[feature.Plot==1].Feature.values
    
    # 计算模型中每个Target Feature在所有Target Feature中的占比
    target_feature_contribution = feature_contribution[feature_contribution["Feature"].isin(target_features)].set_index("Feature").T
    target_feature_contribution["Total"] = list(target_feature_contribution.sum(axis=1).values)
    for feature in target_features:
        target_feature_contribution[feature] = target_feature_contribution[feature] / target_feature_contribution["Total"] * 100
    target_feature_contribution = target_feature_contribution.iloc[:,0:-1].T
    
    # 计算原数据中每个Target Feature在所有Target Feature中的占比
    raw_excel = raw[target_features]
    feature_spd_ratio = pd.DataFrame({"Feature": target_features})

    if N_year == 0:
        total_by_feature = raw_excel.sum().to_frame().T
        total = total_by_feature.T.sum().values[0]
        for j in target_features:
            total_by_feature[j] = total_by_feature[j] / total * 100.0
        feature_spd_ratio["Ratio(%)_all"] = total_by_feature.values[0][:-1]
        
    for i in range(N_year):
        total_by_feature = raw_excel.iloc[-12 * (i+1):,].sum().to_frame().T
        total = total_by_feature.T.sum().values[0]
        for j in target_features:
            total_by_feature[j] = total_by_feature[j] / total * 100.0
        feature_spd_ratio["Ratio(%)_" + "latest " + str(i+1) + " year"] = total_by_feature.values[0]
        
    feature_spd_ratio.set_index("Feature", inplace=True)
    output = target_feature_contribution.merge(feature_spd_ratio, on = "Feature")
    
    # 计算整年份的 模型占比/原数据占比 系数，并且添加一列作为之后作图的label
    if N_year == 0:
        contr = output.iloc[:,0:1].values
        spd = output.iloc[:,1:2].values
        output["Index All"] = contr / spd
        output["Label"] = output.index.map(str) + ": " + round(output["Index All"], 2).map(str)

    for i in range(N_year): 
        contr = output.iloc[:,i:i+1].values
        spd = output.iloc[:,i+N_year:i+1+N_year].values
        col_name = "Index " + str(i+1) + " year"
        output[col_name] = contr / spd
        output["Label " + str(i+1) + " year"] = output.index.map(str) + ": " + round(output[col_name], 2).map(str)
     
    # 绘制散点图并保存
    os.mkdir(f'{path}/Contributions')
    
    if N_year == 0:
        line_range = math.ceil(max(output["Ratio(%)_all"].max(), output["Contribution(%)_all"].max()))
        plt.scatter(output["Ratio(%)_all"], output["Contribution(%)_all"])
        for idx, row in output.iterrows(): 
            plt.text(row["Ratio(%)_all"], row["Contribution(%)_all"], row["Label"])
        plt.plot(range(line_range), range(line_range))
        plt.title("Scatter Plot All data")
        plt.savefig(f"{path}/Contributions/Scatter Plot All data.png", bbox_inches='tight')
        plt.close()

    for i in range(N_year):
        line_range = math.ceil(max(output['Ratio(%)_latest ' + str(i+1) + ' year'].max(), output['Contribution(%)_latest ' + str(i+1) + ' year'].max()))
        plt.scatter(output['Ratio(%)_latest ' + str(i+1) + ' year'], output['Contribution(%)_latest ' + str(i+1) + ' year'])
        for idx, row in output.iterrows(): 
            plt.text(row['Ratio(%)_latest ' + str(i+1) + ' year'], row['Contribution(%)_latest ' + str(i+1) + ' year'], row["Label " + str(i+1) + " year"])
        plt.plot(range(line_range), range(line_range))
        plt.title("Scatter Plot " + str(i+1) + " year")
        plt.savefig(f"{path}/Contributions/Scatter Plot " + str(i+1) + " year.png", bbox_inches='tight')
        plt.close()
    
    writer = pd.ExcelWriter(f"{path}/Contributions/contribution_" + str(data_size) + "M.xlsx")
    contribution.to_excel(writer, sheet_name = "model_data", index = False)
    feature_contribution.to_excel(writer, sheet_name = "all_feature_contribution", index = False)
    output.to_excel(writer, sheet_name = "target_feature_contribution", index = True)
    writer.save()
    print('**********Done Scatter Plotting **********')   

    
#========================= inverse coef =========================# 
def inverse_coef(raw, data_size, feature, fit):
    features = feature.copy()
    features_list = list(features['Feature'].values)
    param_list = fit.get_posterior_mean()
    coefs_list = []
    for coef in param_list:
        mean = coef.mean()
        coefs_list.append(mean)
        
    # 模型系数
    n_coefs = len(features_list) - 1
    coefs = coefs_list[0:n_coefs]
    intercept = coefs_list[n_coefs]
   
    # Specify the time range of the model
    raw = raw.iloc[-data_size:,]
    raw = raw[features_list]
    X = raw.iloc[:,1:]
    y = raw.iloc[:,0]

    # 去标准化后的模型系数
    mcoefs_raw = [coefs[col] * y.std() / X.iloc[:,col].std() for col in range(0,X.shape[1])]

    # 去标准化后的intercept
    mcoefs_raw_intercept = intercept * y.std() + y.mean() - X.dot(mcoefs_raw).mean()
    
    return mcoefs_raw, mcoefs_raw_intercept


#========================= Prediction =========================#
'''目的：把变更小幅度的数据输入到拟合后的模型中做预测
--input:
    · f   被变动的特征名，str
    · delta   变化幅度，float
--output:
'''
# f代表被变动的特征，file是原数据， delta是变化幅度
def predict(f, data, file_name, feature, mode, delta, mcoefs_raw, mcoefs_raw_intercept):  
    
    # 对原数据的指定特征进行幅度为delta的扰动)
    new_data = data.copy()
    new_data[f] = new_data[f] * (1 + delta)
    
    # 对扰动后的数据进行carry over效应的处理
    raw = data_trans.do(new_data, file_name, feature, mode)[1]
    
    # 把X的列的顺序调整到和coef的顺序相同
    features_list = list(feature['Feature'].values)
    raw = raw[features_list]
    
    X = raw.iloc[:, 1:]

    # 去标准化后的prediction
    y_raw_hat = X.dot(mcoefs_raw) + mcoefs_raw_intercept

    # 取均值作为这一组数据的预测
    y_hat = np.average(y_raw_hat)

    # 返回对y的预测，同时返回原数据里的x的均值，作为画图时的横坐标
    return data[f].mean()*(1+delta), y_hat

