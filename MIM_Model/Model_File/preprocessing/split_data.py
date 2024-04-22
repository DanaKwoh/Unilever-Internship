"""
This code page is used to split prediction and features. Only work for the LightGB model. "Target==1" variables are our features, unlike MCMC model. "Target==1" of MCMC means selected features on later scatter plots.
"""

from Model_File.packages import *



def do(data, config):
    date = data.columns[0]
    response = config.Feature[0]
    predict = config.loc[config.Target == 1, 'Feature'].values
    x = data[predict]
    y = data[response]
    return x, y

