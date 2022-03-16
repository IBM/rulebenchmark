from config_copy_copy import config_dict_imbalanced
from config_copy_copy import Config_list
import pandas as pd
import numpy as np
import seaborn as sns
# import os
from sklearn.model_selection import train_test_split 
from sklearn.metrics import matthews_corrcoef,fbeta_score,confusion_matrix,f1_score,precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, r2_score, explained_variance_score, mean_absolute_error, max_error

from sklearn.ensemble import GradientBoostingRegressor
#import wittgenstein as lw
import time
import warnings
import re

from aix360.algorithms.rbm import BRCGExplainer, BooleanRuleCG

from aix360.algorithms.rbm import FeatureBinarizer

from aix360.algorithms.rbm import FeatureBinarizerFromTrees

from aix360i.algorithms.rule_induction.ripper import Ripper

from corels import *

BINARIZER= ['TREES',"QUANTILE","NATIVE"]
ALGO = ['RIPPER',"BRCG","CORELS"]


metric_dict = {}
metric_list = []  

script_time = time.time()
for config in config_dict_imbalanced:
    CONFIG = config_dict_imbalanced[config]
    Pos_class = CONFIG['POS_CLASS'] 
    def convert(char):
        if char == CONFIG['POS_CLASS']:
            return 1
        else:
            return 0

    df = pd.read_csv(CONFIG['DATA_SET'],dtype=CONFIG['DATA_TYPES'])
    df = df.drop(columns=CONFIG['DROP'])
    
    
    print(CONFIG['NAME'])
    df[CONFIG['TARGET_LABEL']].value_counts()

  
   
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=[CONFIG['TARGET_LABEL']]), df[CONFIG['TARGET_LABEL']], test_size=CONFIG['TRAIN_TEST_SPLIT'], random_state=42)

    print('Training:', x_train.shape, y_train.shape)
    print('Test:', x_test.shape, y_test.shape)

    for i in BINARIZER:
        
        
        if i == "TREES":
            binarizer =  FeatureBinarizerFromTrees(negations=True, randomState=42) 
            binarizer = binarizer.fit(x_train, y_train)
            x_train_bin = binarizer.transform(x_train) 
            x_test_bin = binarizer.transform(x_test)
            for algo in ALGO:
                präds = []
                if algo == 'RIPPER':
                    CONFIG['POS_CLASS'] = Pos_class
                    # start time
                    start_time = time.time()
                    try:
                        estimator = Ripper()
                        estimator.fit(x_train_bin, y_train, pos_value=CONFIG['POS_CLASS'])
                        end_time = time.time()
                        y_pred = estimator.predict(x_test_bin)
                        print('------------------------------------------------------') 
                        print('RIPPER TREES')

                        ripper_t_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                        ripper_t_f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
                        ripper_t_acc = accuracy_score(y_test, y_pred)

                        print('Rule count: ' + str(sum([len(rules) for rules in estimator.rule_map.values()])))
                        ripper_t_rl = str(sum([len(rules) for rules in estimator.rule_map.values()]))
                        if ripper_t_rl != 0:
                        
                            for i in range(len(estimator.rule_map[CONFIG['POS_CLASS']])):
                                präds.append(len(estimator.rule_map[CONFIG['POS_CLASS']][i]))
                            ripper_t_präd = sum(präds)
                            ripper_t_max_präd = max(präds)
                        
                        else:
                            ripper_t_präd = 0
                            ripper_t_max_präd= 0
                        print(CONFIG['POS_CLASS'])
                        print('------------------------------------------------------')
                    except Exception:
                        pass
                if algo == 'BRCG':
                    
                    CONFIG['POS_CLASS'] = 1
                    start_time = time.time()
                    estimator = BooleanRuleCG()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        
                        estimator.fit(x_train_bin, y_train)
                        end_time = time.time()
                        y_pred = estimator.predict(x_test_bin)
                    print('------------------------------------------------------') 
                    brcg_t_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                    brcg_t_f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
                    brcg_t_acc = accuracy_score(y_test, y_pred)
                
                    model = estimator.explain()
                    
                    brcg_t_rl = len(model['rules']) 
                    CONFIG['POS_CLASS']
                    print('------------------------------------------------------')

                if algo == 'CORELS':
                    CONFIG['POS_CLASS'] = Pos_class
                    start_time = time.time()
                    estimator = CorelsClassifier(n_iter=10000, 
                        max_card=2, 
                        c = 0.0001 
                        )
                    estimator.fit(x_train_bin, y_train , prediction_name = CONFIG["TARGET_LABEL"])
                    end_time = time.time()
                    y_pred = estimator.predict(x_test_bin)
                    print('------------------------------------------------------') 
                    print('CORELS TREES')

                    corels_t_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                    corels_t_f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
                    corels_t_acc = accuracy_score(y_test, y_pred)
                    r_length = len(estimator.rl().rules)
                    corels_t_rl = len(estimator.rl().rules)
                    print("Rule Length:", r_length)
                    print('------------------------------------------------------')

        if i == "QUANTILE":
            binarizer =  FeatureBinarizer(numThresh=9,negations=True, randomState=42) 
            binarizer = binarizer.fit(x_train)
            x_train_bin = binarizer.transform(x_train) 
            x_test_bin = binarizer.transform(x_test)  

            for algo in ALGO:
                präds = []

                if algo == 'RIPPER':
                    CONFIG['POS_CLASS'] = Pos_class
                    start_time = time.time()
                    try:
                        estimator = Ripper()
                        estimator.fit(x_train_bin, y_train, pos_value=CONFIG['POS_CLASS'])
                        end_time = time.time()
                        y_pred = estimator.predict(x_test_bin)
                        print('------------------------------------------------------') 
                        print('RIPPER QUANTILE')
                        ripper_q_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                        ripper_q_f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
                        ripper_q_acc = accuracy_score(y_test, y_pred)
                        ripper_q_rl = str(sum([len(rules) for rules in estimator.rule_map.values()]))
                        if ripper_q_rl != 0:
                            
                            for i in range(len(estimator.rule_map[CONFIG['POS_CLASS']])):
                                präds.append(len(estimator.rule_map[CONFIG['POS_CLASS']][i]))
                            ripper_q_präd = sum(präds)
                            ripper_q_max_präd = max(präds)

                        else:
                            ripper_q_präd = 0
                            ripper_q_max_präd= 0
                        
                        
                        print('------------------------------------------------------')
                    except Exception:
                        pass
                if algo == 'BRCG':
                    CONFIG['POS_CLASS'] = 1
                    start_time = time.time()
                    estimator = BooleanRuleCG()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        estimator.fit(x_train_bin, y_train)
                        end_time = time.time()
                    y_pred = estimator.predict(x_test_bin)
                    print('------------------------------------------------------')  
                    print('BRCG QUANTILE')
                    
                    brcg_q_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                    brcg_q_f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
                    brcg_q_acc = accuracy_score(y_test, y_pred)
                    
                    model = estimator.explain()
                
        
                    brcg_q_rl = len(model['rules'])
                    print(CONFIG['POS_CLASS'])
                    print('------------------------------------------------------')

                if algo == 'CORELS':
                    CONFIG['POS_CLASS'] = Pos_class
                    start_time = time.time()
                    estimator = CorelsClassifier(n_iter=10000, 
                        max_card=2, 
                        c = 0.0001 
                        )
                    estimator.fit(x_train_bin, y_train , prediction_name = CONFIG["TARGET_LABEL"])
                    end_time = time.time()
                    y_pred = estimator.predict(x_test_bin)
                    print('------------------------------------------------------') 
            
                    corels_q_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                    corels_q_f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
                    corels_q_acc = accuracy_score(y_test, y_pred)
                
                    r_length = len(estimator.rl().rules)
                    corels_q_rl = len(estimator.rl().rules)
            
                    print('------------------------------------------------------')   
                

        if i == "NATIVE":
            CONFIG['POS_CLASS'] = Pos_class
            x_train_bin = x_train
            x_test_bin = x_test

            start_time = time.time()
        
            estimator = Ripper()
            estimator.fit(x_train_bin, y_train, pos_value=CONFIG['POS_CLASS'])
            end_time = time.time()
            y_pred = estimator.predict(x_test_bin)
            print('------------------------------------------------------') 
            print('RIPPER NATIVE')
           
            ripper_n_bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
            ripper_n_f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
            ripper_n_acc = accuracy_score(y_test, y_pred)
            
            print(CONFIG['POS_CLASS'])
            ripper_n_rl = str(sum([len(rules) for rules in estimator.rule_map.values()]))
            if ripper_n_rl != 0:
            
                for i in range(len(estimator.rule_map[CONFIG['POS_CLASS']])):
                    präds.append(len(estimator.rule_map[CONFIG['POS_CLASS']][i]))
                ripper_n_präd = sum(präds)
                ripper_n_max_präd = max(präds)

            else:
                ripper_n_präd = 0
                ripper_n_max_präd= 0
        

              

    
    metric_dict.update({config:{"Config":config,"ripper_t_bacc":ripper_t_bacc, "ripper_t_f2": ripper_t_f2,"ripper_t_acc":ripper_t_acc,"ripper_t_rl":ripper_t_rl,"ripper_t_präd":ripper_t_präd,"ripper_t_max_präd":ripper_t_max_präd, 
                                                "brcg_t_bacc": brcg_t_bacc, "brcg_t_f2": brcg_t_f2,"brcg_t_acc":brcg_t_acc,"brcg_t_rl":brcg_t_rl,
                                                "corels_t_bacc":corels_t_bacc, "corels_t_f2": corels_t_f2,"corels_t_acc":corels_t_acc,"corels_t_rl":corels_t_rl,
                                                "ripper_q_bacc":ripper_q_bacc, "ripper_q_f2":ripper_q_f2,"ripper_q_acc":ripper_q_acc,"ripper_q_rl":ripper_q_rl,"ripper_q_präd":ripper_q_präd,"ripper_q_max_präd":ripper_q_max_präd, 
                                                "brcg_q_bacc":brcg_q_bacc,"brcg_q_f2":brcg_q_f2,"brcg_q_acc":brcg_q_acc,"brcg_q_rl":brcg_q_rl,
                                                "corels_q_bacc":corels_q_bacc,"corels_q_f2":corels_q_f2,"corels_q_acc": corels_q_bacc,"corels_q_rl":corels_q_rl,
                                                "ripper_n_bacc": ripper_n_bacc, "ripper_n_f2": ripper_n_f2,"ripper_n_acc": ripper_n_bacc, "ripper_n_rl":ripper_n_rl,"ripper_n_präd":ripper_n_präd,"ripper_n_max_präd":ripper_n_max_präd}})
    metric_list.append(metric_dict[config])
   
script_time_end = time.time()
print('Training time: ' + str(script_time_end - script_time))

df_list = []
csv_list = []
for i in range(len(Config_list)):
    
    if Config_list[i][1]['TYPE'] == "BINARY":
        if Config_list[i][1]["DATA_SET"] not in csv_list:
            temp_df = pd.read_csv(Config_list[i][1]["DATA_SET"])
            csv_list.append(Config_list[i][1]["DATA_SET"])
            #temp_df = temp_df.drop(columns=Config_list[i][1]['DROP'])
            temp_df= temp_df.rename(columns={temp_df[Config_list[i][1]['TARGET_LABEL']].name : 'TARGET_LABEL'})
            df_list.append(temp_df)
        
eval_df = pd.DataFrame(csv_list, columns=['Data_Set'])
eval_df["Target_1_pos"] = pd.Series('int32')
eval_df["Target_2_neg"] = pd.Series('int32')
eval_df["IB_Ratio"] = pd.Series()
eval_df["Num_Feautures"] = pd.Series('int32')
eval_df["Cat_Feautures"] = pd.Series('int32')
eval_df["Size_row"] = pd.Series('int32')
eval_df["Size_col"] = pd.Series('int32')

# Trees
eval_df["ripper_t_bacc"] = pd.Series()
eval_df["ripper_t_f2"] = pd.Series()
eval_df["ripper_t_acc"] = pd.Series()
eval_df["ripper_t_rl"] = pd.Series()
eval_df["ripper_t_präd"] = pd.Series()
eval_df["ripper_t_max_präd"] = pd.Series()

eval_df["brcg_t_bacc"] = pd.Series()
eval_df["brcg_t_f2"] = pd.Series()
eval_df["brcg_t_acc"] = pd.Series()
eval_df["brcg_t_rl"] = pd.Series()

eval_df["corels_t_bacc"] = pd.Series()
eval_df["corels_t_f2"] = pd.Series()
eval_df["corels_t_acc"] = pd.Series()
eval_df["corels_t_rl"] = pd.Series()

# Quantile
eval_df["ripper_q_bacc"] = pd.Series()
eval_df["ripper_q_f2"] = pd.Series()
eval_df["ripper_q_acc"] = pd.Series()
eval_df["ripper_q_rl"] = pd.Series()
eval_df["ripper_q_präd"] = pd.Series()
eval_df["ripper_q_max_präd"] = pd.Series()

eval_df["brcg_q_bacc"] = pd.Series()
eval_df["brcg_q_f2"] = pd.Series()
eval_df["brcg_q_acc"] = pd.Series()
eval_df["brcg_q_rl"] = pd.Series()

eval_df["corels_q_bacc"] = pd.Series()
eval_df["corels_q_f2"] = pd.Series()
eval_df["corels_q_acc"] = pd.Series()
eval_df["corels_q_rl"] = pd.Series()

# Ripper Native
eval_df["ripper_n_bacc"] = pd.Series()
eval_df["ripper_n_f2"] = pd.Series()
eval_df["ripper_n_acc"] = pd.Series()
eval_df["ripper_n_rl"] = pd.Series()
eval_df["ripper_n_präd"] = pd.Series()
eval_df["ripper_n_max_präd"] = pd.Series()
        
            

for frame in range(len(df_list)):
    
    metric = df_list[frame]["TARGET_LABEL"].value_counts()

    # Imbalanced Ratio = minor class
    if metric[0] > metric[1]:
        eval_df["Target_1_pos"].iloc[frame] = metric[1]
        eval_df["Target_2_neg"].iloc[frame] = metric[0]
    else:
        eval_df["Target_1_pos"].iloc[frame] = metric[0]
        eval_df["Target_2_neg"].iloc[frame] = metric[1]
        
    df_size_row = len(df_list[frame])
    df_size_col = len(df_list[frame].columns)
    df_num_feauture =  len(df_list[frame].select_dtypes(include=['int64', 'float64']).columns)
    df_cat_feauture =   len(df_list[frame].select_dtypes(include=['object']).columns)

    eval_df["IB_Ratio"].iloc[frame] = eval_df["Target_1_pos"].iloc[frame]/eval_df["Target_2_neg"].iloc[frame]
    eval_df["Size_row"].iloc[frame]   = df_size_row
    eval_df["Size_col"].iloc[frame]   = df_size_col
    
    eval_df["Num_Feautures"].iloc[frame] = df_num_feauture
    eval_df["Cat_Feautures"].iloc[frame] = df_cat_feauture

    # adding Metrics Trees
    eval_df["ripper_t_bacc"].iloc[frame] = metric_list[frame]["ripper_t_bacc"]
    eval_df["ripper_t_f2"].iloc[frame] = metric_list[frame]["ripper_t_f2"]
    eval_df["ripper_t_acc"].iloc[frame] = metric_list[frame]["ripper_t_acc"]
    eval_df["ripper_t_rl"].iloc[frame] = metric_list[frame]["ripper_t_rl"]
    eval_df["ripper_t_präd"].iloc[frame] = metric_list[frame]["ripper_t_präd"]
    eval_df["ripper_t_max_präd"].iloc[frame] = metric_list[frame]["ripper_t_max_präd"]
    


    eval_df["brcg_t_bacc"].iloc[frame] = metric_list[frame]["brcg_t_bacc"]
    eval_df["brcg_t_f2"].iloc[frame] = metric_list[frame]["brcg_t_f2"]
    eval_df["brcg_t_acc"].iloc[frame] = metric_list[frame]["brcg_t_acc"]
    eval_df["brcg_t_rl"].iloc[frame] = metric_list[frame]["brcg_t_rl"]

    eval_df["corels_t_bacc"].iloc[frame] = metric_list[frame]["corels_t_bacc"]
    eval_df["corels_t_f2"].iloc[frame] = metric_list[frame]["corels_t_f2"]
    eval_df["corels_t_acc"].iloc[frame] = metric_list[frame]["corels_t_acc"]
    eval_df["corels_t_rl"].iloc[frame] = metric_list[frame]["corels_t_rl"]

    # adding Metrics Qunatile
    eval_df["ripper_q_bacc"].iloc[frame] =  metric_list[frame]["ripper_q_bacc"]
    eval_df["ripper_q_f2"].iloc[frame] = metric_list[frame]["ripper_q_f2"]
    eval_df["ripper_q_acc"].iloc[frame] = metric_list[frame]["ripper_q_acc"]
    eval_df["ripper_q_rl"].iloc[frame] = metric_list[frame]["ripper_q_rl"]
    eval_df["ripper_q_präd"].iloc[frame] = metric_list[frame]["ripper_q_präd"]
    eval_df["ripper_q_max_präd"].iloc[frame] = metric_list[frame]["ripper_q_max_präd"]

    eval_df["brcg_q_bacc"].iloc[frame] = metric_list[frame]["brcg_q_bacc"]
    eval_df["brcg_q_f2"].iloc[frame] = metric_list[frame]["brcg_q_f2"]
    eval_df["brcg_q_acc"].iloc[frame] = metric_list[frame]["brcg_q_acc"]
    eval_df["brcg_q_rl"].iloc[frame] = metric_list[frame]["brcg_q_rl"]
    

    eval_df["corels_q_bacc"].iloc[frame] =  metric_list[frame]["corels_q_bacc"]
    eval_df["corels_q_f2"].iloc[frame] =  metric_list[frame]["corels_q_f2"]
    eval_df["corels_q_acc"].iloc[frame] =  metric_list[frame]["corels_q_acc"]
    eval_df["corels_q_rl"].iloc[frame] =  metric_list[frame]["corels_q_rl"]

    # adding Metrics Ripper Native
    eval_df["ripper_n_bacc"].iloc[frame] = metric_list[frame]["ripper_n_bacc"]
    eval_df["ripper_n_f2"].iloc[frame] = metric_list[frame]["ripper_n_f2"]
    eval_df["ripper_n_acc"].iloc[frame] = metric_list[frame]["ripper_n_acc"]
    eval_df["ripper_n_rl"].iloc[frame] = metric_list[frame]["ripper_n_rl"]
    eval_df["ripper_n_präd"].iloc[frame] = metric_list[frame]["ripper_n_präd"]
    eval_df["ripper_n_max_präd"].iloc[frame] = metric_list[frame]["ripper_n_max_präd"]


eval_df.to_csv("test5.csv",sep =",")