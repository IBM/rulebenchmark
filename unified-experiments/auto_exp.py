from data_configs import CONFIG_DICT_IMBALANCED
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

PIPELINES = [('TREES','BRCG'), ('QUANTILE','BRCG'), 
('TREES','RIPPER'), ('QUANTILE','RIPPER'), ('NATIVE','RIPPER'),('TREES','CORELS'), ('QUANTILE','CORELS')]
TRAIN_TEST_SPLIT =  0.3

result_list = []

script_start_time = time.time()
print('Job started at {}.'.format(time.ctime()))


for config in CONFIG_DICT_IMBALANCED:
    result = {}
    CONFIG = CONFIG_DICT_IMBALANCED[config]
    print('Running data set:', CONFIG['NAME'])
    result['Dataset'] = CONFIG['NAME']

    df_from_file = pd.read_csv(CONFIG['DATA_SET'],dtype=CONFIG['DATA_TYPES'])
    print('Read', len(df_from_file), 'rows from', CONFIG['DATA_SET'])
    result['nof_rows'] = len(df_from_file)

    df_from_file = df_from_file.drop(columns=CONFIG['DROP'])
    
    for (bina, algo) in PIPELINES:
        prefix = bina + '-'  + algo

        df = df_from_file.copy() # at least RIPPER messes with the df, hence we draw a fresh copy

        x_train, x_test, y_train, y_test = train_test_split(
            df.drop(columns=[CONFIG['TARGET_LABEL']]), 
            df[CONFIG['TARGET_LABEL']], 
            test_size=TRAIN_TEST_SPLIT, 
            random_state=42)
        # TODO if we do the split outside this loop we use the same split across pipelines

        print('Training', prefix, 'on', CONFIG['NAME'])
        start_time = time.time()

        # Part 0: Some Preparation
        POS_CLASS = CONFIG['POS_CLASS']
        # BRCG trains for value 1 as POS_CLASS
        if algo == 'BRCG': # or algo == 'CORELS' 
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
        if bina in ['TREES', 'QUANTILE'] and algo == 'RIPPER':
            # RIPPER cannot process multi-index produced by these binarizers, hence flatten multi-index
            x_train_bin = pd.DataFrame(x_train_bin.to_records())
            x_test_bin = pd.DataFrame(x_test_bin.to_records())
            x_train_bin = x_train_bin.drop("index", axis = 1)
            x_test_bin = x_test_bin.drop("index", axis = 1)
            x_train_bin.columns = pd.Index(np.arange(1,len(x_train_bin.columns)+1).astype(str))
            x_test_bin.columns = pd.Index(np.arange(1,len(x_test_bin.columns)+1).astype(str))
        
        # Part 3: Run Rule Induction
        if algo == 'RIPPER':       
            estimator = Ripper()
            estimator.fit(x_train_bin, y_train, pos_value=POS_CLASS)
        elif algo == 'BRCG':                
            estimator = BooleanRuleCG(silent=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")               
                estimator.fit(x_train_bin, y_train)
        elif algo == 'CORELS':
            estimator = CorelsClassifier(n_iter=10000, max_card=2, c = 0.0001, verbosity=[])
            # TODO Why these parameters?
            estimator.fit(x_train_bin, y_train, prediction_name=CONFIG["TARGET_LABEL"])
            # TODO Why do we specify the target label here?

        end_time = time.time()
        training_time = end_time - start_time
        result[prefix+'_runtime'] = training_time
        print('Finished training in {} seconds.'.format(str(training_time)))

        # Part 4: Evaluation
        y_pred = estimator.predict(x_test_bin)
        result[prefix+'_adj_bal_acc'] = balanced_accuracy_score(y_test, y_pred, adjusted=True)
        result[prefix+'_f2'] = fbeta_score(y_test, y_pred, pos_label=POS_CLASS, beta= 2)
        result[prefix+'_acc'] = accuracy_score(y_test, y_pred)

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
        elif algo == 'CORELS':
            # Get predicates
            praed_len = []
            for i in range(len(estimator.rl().rules[0]["antecedents"])):
                praed_len.append(len(estimator.rl().rules[i]["antecedents"]))          
            nof_rules = len(estimator.rl().rules)
            preds_sum = sum(praed_len)
            preds_max = max(praed_len)

        if algo == "RIPPER" or algo == "CORELS":
            result[prefix+'_nof_rules'] = nof_rules
            result[prefix+'_sum_preds'] = preds_sum
            result[prefix+'_max_preds'] = preds_max

    result_list.append(result)

result_df = pd.DataFrame(result_list)
result_df.to_csv('results.csv', sep=',')


    