"""
This code page is used to help set the dataset configures file. It would be required by both models. Please done setting before running the main.py
"""
from Model_File.packages import *

if len(sys.argv) !=2 :
    print('-- Please check the input!!! \n-- It should be like "python; build_config.py; filename".')
    
filename = sys.argv[1]

new = pd.DataFrame(columns = ['Feature', 'Constraints','Target', 'Plot', 'Media', 'CR', 'pw'])
data = pd.read_csv(f'Input/Input_Data/{filename}.csv')
feature = data.columns
new['Feature'] = feature.drop('Date')

print(f'\n>>>>> "{filename}_config.csv" is already created! ! 【This is a configure file for the dataset setting. Plase go to this file "Input/Input_Config/{filename}_config.csv" and define the information.】\n')

new.to_csv(f'Input/Input_Config/{filename}_config.csv', index = 0)   

