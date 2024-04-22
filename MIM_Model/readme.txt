
【"readme.txt": 终端指令输入的格式提醒】
==============================================================
Terminal Input: 三个
==============================================================
>>>>> #1.  自动生成数据congifure表格，并保存至Input/Input_Config, 方便填写参数：
----------------【格式】Format ----------------
    python build_config.py data_filename      
----------------【例子】Example ----------------
    python build_config.py raw_input

>>>>> #2.  自动生成模型congifure表格，并保存至Input/Input_Config, 方便设置MCMC模型参数：
----------------【格式】Format ----------------
    python build_Mconfig.py data_filename n_points data_size   
----------------【例子】Example ----------------
    python build_Mconfig.py raw_input 10 36

>>>>> #3.  最终运行
----------------【格式】Format ---------------- 
    python   main.py  :  fixed
             data filename  :  string (excluding '.csv')
             carry_over method :    -i   OR   -o   OR    -n
             model type:  -L   OR   -M
----------------【例子】Example ----------------
    python main.py raw_input -i -L
==============================================================
Further Improvement:
	None! This is the final version including merging LGBM and MCMC model.
==============================================================


