"""
This code page is used to help setting Model Configures File. Apply on MCMC model only. LGBM does not require this.
"""
from Model_File.packages import *



if len(sys.argv) != 4:
    print('-- Please check the input!!! \n-- It should be like "python; build_Mconfig.py; filename; n_points; data_size".')
    sys.exit()
filename = sys.argv[1]

new = pd.DataFrame(columns = ['Obj', 
                              'ParameterName', 
                              'ParameterValue', 
                              'Explanation'
                             ])
new.Obj = ['TimeSeriesTrans', 
           'MCMC_Model', 
           'MCMC_Model', 
           'MCMC_Model', 
           'MCMC_Model'
          ]
new.ParameterName = ['n_points',
                     'model_name',
                     'stan_name',
                     'data_size',
                     'target_path'
                    ]

new. ParameterValue = [sys.argv[2],
                       'MCMC_V1',
                       'MLR_V1.stan',
                       sys.argv[3],
                       'Target_Feature.csv'
                      ]
new.Explanation = ['ResponseCurve中的描点个数的一半',
                   '自定义的待保存模型pickle文件的名称',
                   'pystan库StanModel所需的一个配置模型参数的.stan文件的名称',
                   '指定训练数据的时间范围（行数）',
                   '进一步需要结合原数据观察贡献度的目标特征'
                  ]
# print(new)  
print(f'\n>>>>> "{filename}_Mconfig.csv" is already created!! Under the folder "Input/Input_Config" 【This is a configure file for the MCMC modelling setting.】\n')
new.to_csv(f'Input/Input_Config/{filename}_MCMC_config.csv', index = 0)   


