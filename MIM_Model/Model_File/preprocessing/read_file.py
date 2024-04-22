"""
从终端中读取输入的指令。并返回相应的数据集、数据参数文件、模型参数文件， 以及carry-over的使用模式。
"""
from Model_File.packages import *


# construct dataframe according to the file's path
def read_file(data_path, config_path):
    raw = pd.read_csv(data_path)
    config = pd.read_csv(config_path)
    return raw, config


path = 'Input/'

# Check the input length 
if len(sys.argv) != 4:
    print("Error on input's length!! Please check the 'README' file.")
    sys.exit()

file_name = sys.argv[1]
data_path = f'{path}Input_Data/{file_name}.csv'
config_path = f'{path}Input_Config/{file_name}_config.csv'
MCMC_config_path = f'{path}Input_Config/{file_name}_MCMC_config.csv'

# Get the carry-over effect mode
mode = sys.argv[2].lower()

#read the raw dataset file and data configures file
raw, config = read_file(data_path, config_path)

# return MCMC model config only
if sys.argv[3] == '-M':
    modelfig = pd.read_csv(MCMC_config_path)


# return the new format of these files
result = [
    raw,
    file_name,
    config,
    mode
    ]

