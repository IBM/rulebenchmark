from data_configs import CONFIG_DICT_IMBALANCED
import pandas as pd
import numpy as np
from statistics import mean


import torch
import numpy as np




import torch
import numpy as np




from sklearn.model_selection import train_test_split 
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score
# matthews_corrcoef, confusion_matrix, f1_score,  r2_score, explained_variance_score, mean_absolute_error, max_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import defaultdict
import re

import time
import warnings
from os.path import exists
import sys

import xgboost as xgb
from aix360.algorithms.rbm import FeatureBinarizer, FeatureBinarizerFromTrees # BRCGExplainer, 
from aix360i.algorithms.rule_induction.rbm.boolean_rule_cg import BooleanRuleCG as BRCG
from aix360i.algorithms.rule_induction.ripper import Ripper
from aix360i.algorithms.rule_induction.r2n.r2n_algo import R2Nalgo
from aix360i.algorithms.rule_induction.r2n.training import train as train_R2N
from corels import *

# Helper Functions
#------------------------------------------------------------------------------------------------------------------#

# For some rule induction algorithms the pos class is always value 1
def convert(char):
    if char == CONFIG['POS_CLASS']:
        return 1
    else:
        return 0

# One Hot Encoding Function
def one_hot_encode_category(df, categorial_index):
                    d = defaultdict(LabelEncoder)
                    # Encoding
                    lecatdata = df[categorial_index].apply(lambda x: d[x.name].fit_transform(x))
                    #One hot encoding with dummy variable
                    dummyvars = pd.get_dummies(df[categorial_index])

                    return dummyvars
# IB Ratio Daniel
def calculate_ib_ratio(target_labels):
    if target_labels[0] > target_labels[1]:
        result['Target_1_pos'] = target_labels[1]
        result['Target_2_neg'] = target_labels[0]
    else:
        result['Target_1_pos'] = target_labels[0]
        result['Target_2_neg'] = target_labels[1]

    return result['Target_1_pos'] / result['Target_2_neg']

def calculate_ib_ratio_pos(target_labels):
    if target_labels[0] == target_labels.loc[CONFIG["POS_CLASS"]]:
        result['Target_1_pos'] = target_labels[0]
        result['Target_2_neg'] = target_labels[1]

    else:
        result['Target_2_neg'] = target_labels[0]
        result['Target_1_pos'] = target_labels[1]

    return result['Target_1_pos'] / result['Target_2_neg']

#------------------------------------------------------------------------------------------------------------------#


PIPELINES = [('XGBPREP','XGBOOST'), ('NATIVE','R2N'), ('TREES','BRCG'), ('QUANTILE','BRCG'), 
('TREES','RIPPER'), ('QUANTILE','RIPPER'), ('NATIVE','RIPPER'), ('TREES','CORELS'), ('QUANTILE','CORELS')]

TRAIN_TEST_SPLIT =  0.3
RESULT_FILE = 'results.csv'

script_start_time = time.time()
print('Job started at {}.'.format(time.ctime()))

if exists(RESULT_FILE):
    result_df = pd.read_csv(RESULT_FILE)
    result_list = result_df.to_dict('records')
else:
    print('No previous results. Starting from scratch.')
    result_list = []
preexisting_datasets = [result['Dataset'] for result in result_list]

# sys.exit()

for index, config in enumerate(CONFIG_DICT_IMBALANCED):   
    CONFIG = CONFIG_DICT_IMBALANCED[config]
    print('Data set {} of {}: {}'.format(index+1, len(CONFIG_DICT_IMBALANCED), CONFIG['NAME']))

    # Check whether data set is already in CSV:
    if  CONFIG['NAME'] not in preexisting_datasets:
        result = {}
        result['Dataset'] = CONFIG['NAME']
        df_from_file = pd.read_csv(CONFIG['DATA_SET'],dtype=CONFIG['DATA_TYPES'])

        # Calculate data set specific metrics
        print('Read', len(df_from_file), 'rows from', CONFIG['DATA_SET'])
        result['nof_rows'] = len(df_from_file)
        result['nof_col'] = len(df_from_file.columns)
        result['nof_num_features'] = len(df_from_file.select_dtypes(include=['int64', 'float64']).columns)
        result['nof_cat_features'] = len(df_from_file.select_dtypes(include=['object']).columns)

        target_labels = df_from_file[CONFIG["TARGET_LABEL"]].value_counts()
        result['IB_ratio_minority'] = calculate_ib_ratio(target_labels)
        result['IB_ratio_pos'] = calculate_ib_ratio_pos(target_labels)

        result['use_Case'] = CONFIG["META_DATA"]["use_case"]
        result['data_flag'] = CONFIG["META_DATA"]["flag"]
        

        df_from_file = df_from_file.drop(columns=CONFIG['DROP'])
        
        for (bina, algo) in PIPELINES:
            prefix = bina + '-'  + algo

            df = df_from_file.copy() # at least RIPPER messes with the df, hence we draw a fresh copy

            # Preprocessing: normalizing data for specific algorithms
            if algo in ('BRCG', 'XGBOOST', 'CORELS', 'R2N'): 
                df[CONFIG['TARGET_LABEL']] = df[CONFIG['TARGET_LABEL']].map(convert)
                POS_CLASS = 1
            else:
                POS_CLASS = CONFIG['POS_CLASS']

            # If XGboost we need the categorical and numerical data for the one hot Encoding function
            if algo == "XGBOOST":
                categorial_feat = df.select_dtypes(include=['object']).columns
                numercial_feat = df.select_dtypes(include=['int64', 'float64']).columns
                if categorial_feat.empty:
                    df = df
                else:
                    dummyvars = one_hot_encode_category(df,categorial_feat)
                    df = pd.concat([df[numercial_feat], dummyvars], axis = 1)

            x_train, x_test, y_train, y_test = train_test_split(
                df.drop(columns=[CONFIG['TARGET_LABEL']]), 
                df[CONFIG['TARGET_LABEL']], 
                test_size=TRAIN_TEST_SPLIT, 
                random_state=42)
            

            print('Training', prefix, 'on', CONFIG['NAME'])
            start_time = time.time()

            # Part 1: Run Binarizer
            if bina == 'XGBPREP':
                x_train_bin = x_train
                x_test_bin = x_test
                # Feauture Names are not allowed to contain , or < and must be strings
                regex = re.compile(r"\[|\]|<", re.IGNORECASE)
                x_train_bin.columns = [regex.sub("_", str(col)) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_train_bin.columns.values]
                x_test_bin.columns = [regex.sub("_", str(col)) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_test_bin.columns.values]


                
            elif bina == "TREES":
                binarizer =  FeatureBinarizerFromTrees(negations=True, randomState=42) 
                binarizer = binarizer.fit(x_train, y_train)
                x_train_bin = binarizer.transform(x_train) 
                x_test_bin = binarizer.transform(x_test)
                result[prefix + '_train_bina_cols'] = len(x_train_bin.columns)
            elif bina == "QUANTILE":
                binarizer =  FeatureBinarizer(numThresh=9,negations=True, randomState=42) 
                binarizer = binarizer.fit(x_train)
                x_train_bin = binarizer.transform(x_train) 
                x_test_bin = binarizer.transform(x_test)
                result[prefix + '_train_bina_cols'] = len(x_train_bin.columns)
            elif bina == "NATIVE":
                x_train_bin = x_train
                x_test_bin = x_test
                result[prefix + '_train_bina_cols'] = len(x_train_bin.columns)

            # Part 2: Adapter: Binarizer -> Rule Induction
            if bina in ['TREES', 'QUANTILE'] and algo == 'RIPPER':
                # RIPPER cannot process multi-index produced by these binarizers, hence flatten multi-index
                x_train_bin = pd.DataFrame(x_train_bin.to_records())
                x_test_bin = pd.DataFrame(x_test_bin.to_records())
                x_train_bin = x_train_bin.drop("index", axis = 1)
                x_test_bin = x_test_bin.drop("index", axis = 1)
                x_train_bin.columns = pd.Index(np.arange(1,len(x_train_bin.columns)+1).astype(str))
                x_test_bin.columns = pd.Index(np.arange(1,len(x_test_bin.columns)+1).astype(str))
            
            # Part 3: Run Rule Induction 
            training_exception = False

            if algo == 'XGBOOST':        
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
                    estimator = R2Nalgo(n_seeds=2, max_epochs=5*10**2, min_temp = 10**-4, decay_rate=0.98, coef = 5*10**-4, normalize_num=True,negation=False)
                    estimator.fit(x_train_bin, y_train)
                except Exception:
                    exception_caught = True
            

            end_time = time.time()
            training_time = end_time - start_time
            if training_exception:
                print('Training failed.')
                result[prefix+'_runtime'] = np.nan
            else:
                print('Finished training successfully in {} seconds.'.format(str(training_time)))
                result[prefix+'_runtime'] = training_time


            # Part 4: Evaluation
            prediction_exception = False
            if not training_exception:
                try:
                    y_pred = estimator.predict(x_test_bin)
                except Exception:
                    print('\t Exception in prediction')
                    prediction_exception = True
            if not (training_exception or prediction_exception):
    
                result[prefix+'_acc'] = accuracy_score(y_test, y_pred)
                result[prefix+'_adj_bal_acc'] = balanced_accuracy_score(y_test, y_pred, adjusted=True)
                result[prefix+'_recall'] = recall_score(y_test, y_pred, pos_label=POS_CLASS)
                result[prefix+'_precision'] = precision_score(y_test, y_pred, pos_label=POS_CLASS)
                result[prefix+'_f2'] = fbeta_score(y_test, y_pred, pos_label=POS_CLASS, beta= 2)
            else:
                result[prefix+'_adj_bal_acc'] = np.nan
                result[prefix+'_f2'] = np.nan
                result[prefix+'_acc'] = np.nan
                result[prefix+'_recall'] = np.nan
                result[prefix+'_precision'] = np.nan


            if algo in ('RIPPER', 'BRCG', 'R2N'):
                export_exception = False
                if algo in ('RIPPER', 'BRCG'):
                    rule_set = estimator.explain()
                else:
                    try:
                        rule_set = estimator.export_rules_to_trxf_dnf_ruleset()
                    except Exception:
                        print('\t Exception in trxf export.')
                        export_exception = True
                if not export_exception:
                    conjunctions = rule_set.list_conjunctions()
                    nof_rules = len(conjunctions)
                    conjunction_len = [conjunction.len() for conjunction in conjunctions]
                    preds_sum = sum(conjunction_len)
                    preds_max = max(conjunction_len, default=0)
                    if nof_rules == 0:
                        preds_avg = 0
                    else:
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



            if algo in ('RIPPER', 'BRCG', 'R2N', 'CORELS') and not(training_exception or prediction_exception or export_exception):
                result[prefix+'_nof_rules'] = nof_rules
                result[prefix+'_sum_preds'] = preds_sum
                result[prefix+'_max_preds'] = preds_max
                result[prefix+'_avg_preds'] = preds_avg
                result[prefix+'_combined_size'] = nof_rules + preds_sum

            # TODO Add R2N trxf output (when fixed)

        result_list.append(result)

    else:
        print('Results precomputed.')

    if index == 14:
        break
        # Use for testing

result_df = pd.DataFrame(result_list)
result_df.to_csv(RESULT_FILE, sep=',', index=False)

script_end_time = time.time()
script_time = script_end_time - script_start_time
print('Job finished at {} in {} seconds.'.format(time.ctime(), script_time))