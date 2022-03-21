from data_configs import CONFIG_DICT_IMBALANCED
import pandas as pd
import numpy as np
from statistics import mean

from sklearn.model_selection import train_test_split 
from sklearn.metrics import matthews_corrcoef,fbeta_score,confusion_matrix,f1_score,precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, r2_score, explained_variance_score, mean_absolute_error, max_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
import re

import aix360i.algorithms.rule_induction.r2n.r2n_algo as algo
from aix360i.algorithms.rule_induction.r2n.training import train as train_R2N

import time
import warnings


from aix360.algorithms.rbm import FeatureBinarizer, FeatureBinarizerFromTrees # BRCGExplainer, 
from aix360i.algorithms.rule_induction.rbm.boolean_rule_cg import BooleanRuleCG as BRCG
from aix360i.algorithms.rule_induction.ripper import Ripper
from corels import *
#import wittgenstein as lw


# For some rule induction algorithms the pos class is always value 1
def convert(char):
    if char == CONFIG['POS_CLASS']:
        return 1
    else:
        return 0

PIPELINES = [('NATIVE','XGBOOST'),('NATIVE','R2N'),('TREES','R2N'),('QUANTILE','R2N'),('TREES','BRCG'), ('QUANTILE','BRCG'), 
('TREES','RIPPER'), ('QUANTILE','RIPPER'), ('NATIVE','RIPPER'),('TREES','CORELS'), ('QUANTILE','CORELS')]
TRAIN_TEST_SPLIT =  0.3

result_list = []

script_start_time = time.time()
print('Job started at {}.'.format(time.ctime()))
df_name_check = pd.read_csv("results.csv")
df_name_check = df_name_check.drop("Unnamed: 0", axis = 1)

for config in CONFIG_DICT_IMBALANCED:
    

        
    
    CONFIG = CONFIG_DICT_IMBALANCED[config]

    # Check whether Dataset is already in CSV:
    if  df_name_check.empty or CONFIG['NAME'] not in df_name_check["Dataset"].unique():
        result = {}
        print('Running data set:', CONFIG['NAME'])
        result['Dataset'] = CONFIG['NAME']

        df_from_file = pd.read_csv(CONFIG['DATA_SET'],dtype=CONFIG['DATA_TYPES'])

        # Calcualte Dataset specific Metrics
        print('Read', len(df_from_file), 'rows from', CONFIG['DATA_SET'])
        result['nof_rows'] = len(df_from_file)
        result['nof_col'] = len(df_from_file.columns)
        result['nof_num_features'] = len(df_from_file.select_dtypes(include=['int64', 'float64']).columns)
        result['nof_cat_features'] = len(df_from_file.select_dtypes(include=['object']).columns)
        target_labels = df_from_file[CONFIG["TARGET_LABEL"]].value_counts()
        if target_labels[0] > target_labels[1]:
            result['Target_1_pos'] = target_labels[1]
            result['Target_2_neg'] = target_labels[0]
        else:
            result['Target_1_pos'] = target_labels[0]
            result['Target_2_neg'] = target_labels[1]
        
        result['IB_Ratio'] = result['Target_1_pos'] / result['Target_2_neg']


        df_from_file = df_from_file.drop(columns=CONFIG['DROP'])
        
        for (bina, algo) in PIPELINES:
            prefix = bina + '-'  + algo

            df = df_from_file.copy() # at least RIPPER messes with the df, hence we draw a fresh copy
            

            x_train, x_test, y_train, y_test = train_test_split(
                df.drop(columns=[CONFIG['TARGET_LABEL']]), 
                df[CONFIG['TARGET_LABEL']], 
                test_size=TRAIN_TEST_SPLIT, 
                random_state=42)
            

            print('Training', prefix, 'on', CONFIG['NAME'])
            start_time = time.time()

            # Part 0: Some Preparation
            POS_CLASS = CONFIG['POS_CLASS']
            # BRCG trains for value 1 as POS_CLASS
            if algo == 'BRCG' or algo =="XGBOOST": # or algo == 'CORELS' 
                df[CONFIG['TARGET_LABEL']] = df[CONFIG['TARGET_LABEL']].map(convert)
                POS_CLASS = 1

            # Part 1: Run Binarizer
            if bina == "TREES":
                binarizer =  FeatureBinarizerFromTrees(negations=True, randomState=42) 
                binarizer = binarizer.fit(x_train, y_train)
                x_train_bin = binarizer.transform(x_train) 
                x_test_bin = binarizer.transform(x_test)
            elif bina == "QUANTILE":
                binarizer =  FeatureBinarizer(numThresh=9,negations=True, randomState=42) 
                binarizer = binarizer.fit(x_train)
                x_train_bin = binarizer.transform(x_train) 
                x_test_bin = binarizer.transform(x_test)
            elif bina == "NATIVE":
                x_train_bin = x_train
                x_test_bin = x_test

            # Part 2: Adapter: Binarizer -> Rule Induction
            if bina in ['TREES', 'QUANTILE','NATIVE'] and algo == 'RIPPER':
                # RIPPER cannot process multi-index produced by these binarizers, hence flatten multi-index
                x_train_bin = pd.DataFrame(x_train_bin.to_records())
                x_test_bin = pd.DataFrame(x_test_bin.to_records())
                x_train_bin = x_train_bin.drop("index", axis = 1)
                x_test_bin = x_test_bin.drop("index", axis = 1)
                x_train_bin.columns = pd.Index(np.arange(1,len(x_train_bin.columns)+1).astype(str))
                x_test_bin.columns = pd.Index(np.arange(1,len(x_test_bin.columns)+1).astype(str))
            

            # Part 3: Run Rule Induction
            if algo == 'XGBOOST':       
        
            
                categorical_features = x_train_bin.select_dtypes(include=['object']).columns
                print(categorical_features)

                for col in categorical_features:
                    label_encoder = LabelEncoder()
                    label_encoder = label_encoder.fit(df[col])
                    x_train_bin[col] = label_encoder.transform(x_train_bin[col])
                    x_test_bin[col] = label_encoder.transform(x_test_bin[col])

                estimator = xgb.XGBClassifier(use_label_encoder=False)
                estimator.fit(x_train_bin, y_train)

            elif algo == 'RIPPER':       
                estimator = Ripper()
                estimator.fit(x_train_bin, y_train, pos_value=POS_CLASS)
            elif algo == 'BRCG':   
                
                estimator = BRCG(silent=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")               
                    estimator.fit(x_train_bin, y_train)
            elif algo == 'CORELS':
                estimator = CorelsClassifier(verbosity=[]) # n_iter=10000, max_card=2, c = 0.0001, 
                estimator.fit(x_train_bin, y_train, prediction_name=CONFIG["TARGET_LABEL"])
                
            elif algo == 'R2N':
                try:
                    estimator = algo.R2Nalgo(n_seeds=3, max_epochs=100, decay_rate=0.998, coef = 10**-3, normalize_num=True)
                    estimator.fit(x_train_bin, y_train)
                except Exception:
                    print("Exception occured")
                    pass


            end_time = time.time()
            training_time = end_time - start_time
            result[prefix+'_runtime'] = training_time
            print('Finished training in {} seconds.'.format(str(training_time)))

            # Part 4: Evaluation
            try:
                y_pred = estimator.predict(x_test_bin)
                result[prefix+'_adj_bal_acc'] = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                result[prefix+'_f2'] = fbeta_score(y_test, y_pred, pos_label=POS_CLASS, beta= 2)
                result[prefix+'_acc'] = accuracy_score(y_test, y_pred)
                result[prefix+'_recall'] = recall_score(y_test, y_pred, pos_label=POS_CLASS)
                result[prefix+'_precision'] = precision_score(y_test, y_pred, pos_label=POS_CLASS)
            except Exception:
                result[prefix+'_adj_bal_acc'] = 0
                result[prefix+'_f2'] = 0
                result[prefix+'_acc'] = 0
                result[prefix+'_recall'] = 0
                result[prefix+'_precision'] = 0
                pass

            if algo == 'RIPPER':
                if len(estimator.rule_map.values()) == 0:
                    # TODO this is a RIPPER trxf export bug that needs to be fixed
                    nof_rules = 0
                    preds_sum = 0
                    preds_max = 0
                else:
                    rule_set = estimator.export_rules_to_trxf_dnf_ruleset(POS_CLASS)
                    conjunctions = rule_set.list_conjunctions()
                    nof_rules = len(conjunctions)
                    conjunction_len = [conjunction.len() for conjunction in conjunctions]
                    preds_sum = sum(conjunction_len)
                    preds_max = max(conjunction_len)
                    preds_avg = preds_sum/nof_rules

            elif algo == 'BRCG':
            
                rule_set  = estimator.explain()
                conjunctions = rule_set.conjunctions
                if len(conjunctions) == 0:
                    nof_rules = 0
                    preds_sum = 0
                    preds_max = 0
                else:
                    
                    conjunction_len = [conjunction.len() for conjunction in conjunctions]
                    nof_rules = len(conjunctions)
                    preds_sum = sum(conjunction_len)
                    preds_max = max(conjunction_len)
                    preds_avg = preds_sum/nof_rules

            elif algo == 'CORELS':
                # Get predicates
                praed_len = []
                for i in range(len(estimator.rl().rules[0]["antecedents"])):
                    praed_len.append(len(estimator.rl().rules[i]["antecedents"]))          
                nof_rules = len(estimator.rl().rules)
                preds_sum = sum(praed_len)
                preds_max = max(praed_len)
                preds_avg = preds_sum/nof_rules
                

            if algo == "RIPPER" or algo == "CORELS" or algo == "BRCG":
                result[prefix+'_nof_rules'] = nof_rules
                result[prefix+'_sum_preds'] = preds_sum
                result[prefix+'_max_preds'] = preds_max
                result[prefix+'_model_complexity'] = nof_rules + preds_sum
                result[prefix+'_rule_complexity'] = preds_avg

        result_list.append(result)
        if df_name_check.empty == False:
            new_config = pd.DataFrame(result_list)
            df_name_check = pd.concat([df_name_check, new_config], ignore_index=True)

    else:
        pass
    

    if df_name_check.empty:
        result_df = pd.DataFrame(result_list)
        result_df.to_csv('results.csv', sep=',')
    else:
        df_name_check.to_csv("results.csv", sep = ',')


    