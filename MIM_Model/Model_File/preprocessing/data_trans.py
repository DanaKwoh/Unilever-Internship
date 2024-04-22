"""
功能：基于原数据及配置文件(Configures File)，对媒体相关特征进行时间序列转换
    1. filename_config.csv，特征配置文件
    2. Target 训练使用到的特征
    4. Media 媒体变量特征
    3. CR 系数
    4. pw 幂函数的指数。
    5. 转换模式。-i，对每一项进行pow转换；-o，线性转换完毕后对整列进行pow转换；-n，无pow转换，只进行CR转换。

模型输出：
    1. Input_Data文件夹里会生成后缀名为所用系数及相应转换模式的输出文件，可作为新数据文件使用。
"""
from Model_File.packages import *


def linearTransCol(data, colName, CR):
    new_list = []
    colData = data[colName].to_frame()
    for index, row in colData.iterrows():
        if index == 0:
            new_list.append(row[colName])
            temp_last = row[colName]
        else:
            new = row[colName] + CR * temp_last  
            temp_last = new
            new_list.append(new)
    return new_list
    
def linearTransCol_POW(data, colName, CR, pw):
    new_list = []
    colData = data[colName].to_frame()
    for index, row in colData.iterrows():
        if index == 0:
            if row[colName] < 0: # 对于POW操作，若数值小于0，全部转为0
                val = pow(0, pw)
            else: 
                val = pow(row[colName], pw)
            new_list.append(val)
            temp_last = val
        else:
            if row[colName] < 0: # 对于POW操作，若数值小于0，全部转为0
                new = pow(0 + CR * temp_last, pw)
            else: 
                new = pow(row[colName] + CR * temp_last, pw)
            temp_last = new
            new_list.append(new)
    return new_list
        
#========================= Mode 1 (Outer POW) =========================#
def mode1(data, file_name, media): # 所有数据进行CR转换之后，对转换后的整列进行POW转
    new_DF = pd.DataFrame()    
    for colName in media.Feature.values:
        CR = media.loc[media.Feature == colName, 'CR'].values[0]
        pw = media.loc[media.Feature == colName, 'pw'].values[0]
        new_list = linearTransCol(data, colName, CR)
        new_DF[colName] = pow(pd.DataFrame(new_list), pw)     
    data_wo_media = data.drop(columns = media.Feature.values)
    new_input = pd.merge(data_wo_media, new_DF, how = 'outer', right_index = True, left_index = True)
    path = f"Input/Input_Data/{file_name}_outerPOW.csv"
    return path, new_input

#========================= Mode 2 (Inner POW) =========================#
def mode2(data, file_name, media): # 对于每一行，进行CR与POW转换(Preferred)
    new_DF = pd.DataFrame()
    for colName in media.Feature.values:
        CR = media.loc[media.Feature == colName, 'CR'].values[0]
        pw = media.loc[media.Feature == colName, 'pw'].values[0]
        new_list = linearTransCol_POW(data, colName, CR, pw)  
        new_DF[colName] = new_list   
    data_wo_media = data.drop(columns = media.Feature.values)
    new_input = pd.merge(data_wo_media, new_DF, how = 'outer', right_index = True, left_index = True)
    path = f"Input/Input_Data/{file_name}_innerPOW.csv"
    return path, new_input

#========================= Mode 3 (No POW) =========================#
def mode3(data, file_name, media): # 只对于每一行进行CR转换
    new_DF = pd.DataFrame()
    for colName in media.Feature.values:
        CR = media.loc[media.Feature == colName, 'CR'].values[0]
        new_list = linearTransCol(data, colName, CR)
        new_DF[colName] = new_list        
    data_wo_media = data.drop(columns = media.Feature.values)
    new_input = pd.merge(data_wo_media, new_DF, how = 'outer', right_index = True, left_index = True)
    path = f"Input/Input_Data/{file_name}_noPOW.csv"
    return path, new_input




#========================= Main Method =========================#
def do(data, file_name, config, mode):
    media = config.loc[(config.Media == 1) & (config.Target == 1)]
    # check carry-over method
    if mode == "-o":
        path, new_input = mode1(data, file_name, media)
    if mode == "-i":
        path, new_input = mode2(data, file_name, media)
    if mode == "-n":
        path, new_input = mode3(data, file_name, media)
#     print(f"--------------- Done Transformation -----------------\n\n\n")

    return path, new_input #return dataframe of new data & path of saved file
   



