# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:42:44 2021
@author: Ziheng Yang
"""

"""
功能：基于配置文件，生成pystan库训练MCMC模型所需的.stan文件。作为被MCMC_Model.py所调用的模块。
"""

import pandas as pd

def generate(features, stan_name):
 
    # Get Selected Features List and corresponding constraints list
    features_list = list(features['Feature'].values)
    
    # Format the feature names for generating the stan file
    feature_list_stan = [i.replace("&", "").replace("-", "_").replace(".", "_") for i in features_list]
    constraint_list_stan = list(features['Constraints'].values)
    
    # Write to Stan file
    with open('Output/Output_StanFiles/' + stan_name, 'w') as f:
        
        # Data
        f.writelines(["data {\n", "\tint<lower=0> N;\n"])
        for i in feature_list_stan:
            f.writelines(["\tvector[N] ", i, ";\n"])
        f.writelines(["}\n\n"])
        
        # Parameters
        f.writelines(["parameters {\n"])
        for i, val in enumerate(feature_list_stan):
            if i != 0:
                constraint = constraint_list_stan[i]
                if constraint == 1:
                    f.writelines(["\treal<lower=0> ", "b_", val, ";\n"])
                elif constraint == -1:
                    f.writelines(["\treal<upper=0> ", "b_", val, ";\n"])
                else:
                    f.writelines(["\treal ", "b_", val, ";\n"])
        f.writelines(["\n\treal beta;\n"])
        f.writelines(["\treal<lower=0> sigma;\n"])
        f.writelines(["}\n\n"])
        
        # Model
        f.writelines(["model {\n"])
        for i, val in enumerate(feature_list_stan):
            if i != 0:
                f.writelines(["\tb_", val, " ~ normal(0, 1);\n"])
                
        f.writelines(["\n\tbeta ~ normal(0, 1);\n"])
        f.writelines(["\tsigma ~ cauchy(0, 5);\n\n"])
        
        # Formula 1
        f.writelines(["\t", feature_list_stan[0], " ~ normal(beta"])
        for i, val in enumerate(feature_list_stan):
            if i != 0:
                f.writelines([" + b_", val, " * ", val])
        f.writelines([", sigma);\n"])
        f.writelines(["}\n\n"])
        
        # Generated Quantities
        f.writelines(["generated quantities {\n", "\treal prediction[N];\n", "\treal log_lik[N];\n", ])
        f.writelines(["\tfor (n in 1:N)\n", "\t\tprediction[n] = normal_rng(beta"])
        for i, val in enumerate(feature_list_stan):
            if i != 0:
                f.writelines([" + b_", val, " * ", val, "[n]"])
        f.writelines([", sigma);\n\n"])
        
        f.writelines(["\tfor (n in 1:N)\n", "\t\tlog_lik[n] = normal_lpdf(", feature_list_stan[0], "[n] | beta"])
        for i, val in enumerate(feature_list_stan):
            if i != 0:
                f.writelines([" + b_", val, " * ", val, "[n]"])
        f.writelines([", sigma);\n\n"])
        f.writelines(["}\n\n"])