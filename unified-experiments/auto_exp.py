from config_copy_copy import CONFIG_DICT_IMBALANCED, CONFIG_LIST
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.metrics import matthews_corrcoef,fbeta_score,confusion_matrix,f1_score,precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, r2_score, explained_variance_score, mean_absolute_error, max_error
from sklearn.ensemble import GradientBoostingRegressor

import time
import warnings

from aix360.algorithms.rbm import BooleanRuleCG, FeatureBinarizer, FeatureBinarizerFromTrees # BRCGExplainer, 
from aix360i.algorithms.rule_induction.ripper import Ripper
from corels import *
#import wittgenstein as lw


# For some rule induction algorithms the pos class is always value 1
def convert(char):
    if char == CONFIG['POS_CLASS']:
        return 1
    else:
        return 0

BINARIZER = ['TREES',"QUANTILE","NATIVE"]
ALGO = ['RIPPER',"BRCG","CORELS"]

PIPELINES = [('TREES','RIPPER'), ('QUANTILE','RIPPER'), ('NATIVE','RIPPER'), 
('TREES','BRCG'), ('QUANTILE','BRCG'), ('TREES','CORELS'), ('QUANTILE','CORELS')]

TRAIN_TEST_SPLIT =  0.3


metric_dict = {}
metric_list = []

result_list = []

script_time = time.time()
print('Starting job at', script_time)


for config in CONFIG_DICT_IMBALANCED:
    result = {}
    CONFIG = CONFIG_DICT_IMBALANCED[config]
    print('Running data set:', CONFIG['NAME'])
    result['Dataset'] = CONFIG['NAME']
    Pos_class = CONFIG['POS_CLASS'] 

    df = pd.read_csv(CONFIG['DATA_SET'],dtype=CONFIG['DATA_TYPES'])
    print('Read', len(df), 'rows from', CONFIG['DATA_SET'])
    result['NO_Rows'] = len(df)

    df = df.drop(columns=CONFIG['DROP'])
    
    print(df[CONFIG['TARGET_LABEL']].value_counts())

    x_train, x_test, y_train, y_test = train_test_split(
        df.drop(columns=[CONFIG['TARGET_LABEL']]), 
        df[CONFIG['TARGET_LABEL']], 
        test_size=TRAIN_TEST_SPLIT, 
        random_state=42)

    print('Training:', x_train.shape, y_train.shape)
    print('Test:', x_test.shape, y_test.shape)

    for (bina, algo) in PIPELINES:
        prefix = bina + '-'  + algo
        if bina == "TREES":
            binarizer =  FeatureBinarizerFromTrees(negations=True, randomState=42) 
            binarizer = binarizer.fit(x_train, y_train)
            x_train_bin = binarizer.transform(x_train) 
            x_test_bin = binarizer.transform(x_test)
        
        
            if algo == 'RIPPER':
                x_train_bin = pd.DataFrame(x_train_bin.to_records())
                x_test_bin = pd.DataFrame(x_test_bin.to_records())
                x_train_bin = x_train_bin.drop("index", axis = 1)
                x_test_bin = x_test_bin.drop("index", axis = 1)
                x_train_bin.columns = pd.Index(np.arange(1,len(x_train_bin.columns)+1).astype(str))
                x_test_bin.columns = pd.Index(np.arange(1,len(x_test_bin.columns)+1).astype(str))
                CONFIG['POS_CLASS'] = Pos_class
                # start time
                start_time = time.time()
                
                estimator = Ripper()
                estimator.fit(x_train_bin, y_train, pos_value=CONFIG['POS_CLASS'])
                end_time = time.time()
                y_pred = estimator.predict(x_test_bin)

                
                print('------------------------------------------------------') 
                print('RIPPER TREES')
                    
            if algo == 'BRCG':
                    
                CONFIG['POS_CLASS'] = 1
                start_time = time.time()
                estimator = BooleanRuleCG()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    estimator.fit(x_train_bin, y_train)
                    end_time = time.time()
                    y_pred = estimator.predict(x_test_bin)

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
    
            
        if bina == "QUANTILE":
            binarizer =  FeatureBinarizer(numThresh=9,negations=True, randomState=42) 
            binarizer = binarizer.fit(x_train)
            x_train_bin = binarizer.transform(x_train) 
            x_test_bin = binarizer.transform(x_test)
            

            if algo == 'RIPPER':

                x_train_bin = pd.DataFrame(x_train_bin.to_records())
                x_test_bin = pd.DataFrame(x_test_bin.to_records())
                x_train_bin = x_train_bin.drop("index", axis = 1)
                x_test_bin = x_test_bin.drop("index", axis = 1)
                x_train_bin.columns = pd.Index(np.arange(1,len(x_train_bin.columns)+1).astype(str))
                x_test_bin.columns = pd.Index(np.arange(1,len(x_test_bin.columns)+1).astype(str))
                CONFIG['POS_CLASS'] = Pos_class
                # start time
                start_time = time.time()
                
                estimator = Ripper()
                estimator.fit(x_train_bin, y_train, pos_value=CONFIG['POS_CLASS'])
                end_time = time.time()
                y_pred = estimator.predict(x_test_bin)

                
                print('------------------------------------------------------') 
                print('RIPPER QUANTILE')
                    
            if algo == 'BRCG':
                    
                CONFIG['POS_CLASS'] = 1
                start_time = time.time()
                estimator = BooleanRuleCG()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    estimator.fit(x_train_bin, y_train)
                    end_time = time.time()
                    y_pred = estimator.predict(x_test_bin)

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

            if bina == "NATIVE":
                binarizer =  FeatureBinarizer(numThresh=9,negations=True, randomState=42) 
                binarizer = binarizer.fit(x_train)
                x_train_bin = x_train
                x_test_bin = x_test

        if algo == 'RIPPER':
            try:
                
                rule_set_list = []
                rule_set = estimator.export_rules_to_trxf_dnf_ruleset(CONFIG['POS_CLASS'])
                conjunctions = rule_set.conjunctions
                praed_len = []
                for c in conjunctions:
                    conjunction_dict = {}
                    predicates = c.predicates
                    praed_len.append(len(predicates))
                    for p in predicates:
                        name = str(p.feature) + str(p.relation)
                        value = p.value
                        conjunction_dict[name] = value
                
                rule_set_list.append(conjunction_dict)
                ruleset_length = str(sum([len(rules) for rules in estimator.rule_map.values()]))
                praed_sum = sum(praed_len)
                max_rule_praed = max(praed_len)
            except Exception:
                ruleset_length = 0
                max_rule_praed = 0
                praed_sum = 0
        
        
    
        if algo == 'CORELS':
        # Get predicates
            praed_len = []
            for i in range(len(estimator.rl().rules[0]["antecedents"])):
                praed_len.append(len(estimator.rl().rules[i]["antecedents"]))
                
            ruleset_length = len(estimator.rl().rules)
            praed_sum = sum(praed_len)
            max_rule_praed = max(praed_len)

    
        


        bacc = balanced_accuracy_score(y_test, y_pred, adjusted=True)
        f2 = fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2)
        acc = accuracy_score(y_test, y_pred)
        # run binarizer
        # run algo
        # compute metrics
        result[prefix+'_bacc'] = bacc
        result[prefix+'_f2'] = f2
        result[prefix+'_acc'] = acc
        if algo == "RIPPER" or algo == "CORELS":
            result[prefix+'_rule_length'] = ruleset_length
            result[prefix+'_sum_praeds'] = praed_sum
            result[prefix+'_max_rule_praed'] = max_rule_praed


    result_list.append(result)
    result_df = pd.DataFrame(result_list)
    result_df.to_csv('results.csv', sep=',')


    