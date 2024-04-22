from Model_File.packages import *
from Model_File.preprocessing import data_trans

 #------------------ Calculation --------------------

# compute the elasticity of one given feature
def elasticity(feature, x, y, model):
    y_old = model.predict(x) #old y's mean given by the model
    y_change = [[]]*2 
    E = []
    interval = 0.05
    #calculate the slope (our elasticisity) in the range of [0.95X, 1.05X]
    x_new = [x.copy(), x.copy()]
    for i in range(2): #so we have 1+0.5 and 1-0.5 percentage.
        x_new[i][feature] *= 1+interval*((-1)**(i+1))
        y_new = model.predict(x_new[i]) #new y's mean
        y_change[i] = (y_new.mean() - y_old.mean()) / y_old.mean()        
        e = y_change[i] / interval*((-1)**(i+1))
        E.append((e*5).round(4)) #elasticity times 5
    return E



# compute the mean absolute precentage error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



__all__ = [
    'elasticity',
    'mean_absolute_percentage_error'
    ]


