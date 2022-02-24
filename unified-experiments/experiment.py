# %% [markdown]
# # Generic Rule Induction Notebook
# 
# Continuoulsy refined.

# %% [markdown]
# ## README
# 
# - GLRM needs to run in proper (conda) aix360 environment
# - BRCG runs with proper (conda) aix360 environment
# - Use aix360i environment for RIPPER

# %% [markdown]
# ### Configuration
# 

# %%

CONFIG1 = { 
    'NAME' : 'binary-churn-quantile-brcg', 
    'DATA_SET': '../data/churn_prob_out_35.csv',
    'DATA_TYPES': {'Children': float, 'RatePlan': str},
    'DROP': ['Id', 'pChurn', '3_Class', '5_Class', 'is_test_set'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3, # 'FIXED' for using 'is_test_set'
    'BINARIZER': 'QUANTILE',
    'ALGO': 'BRCG',
    'TARGET_LABEL': 'CHURN',
    'TYPE' : 'BINARY',
    'EXAMPLE_FEATURE' : 'Est Income',
    'POS_CLASS': 'T',
    'BASELINE': True
     }
CONFIG2 = { 
    'NAME' : 'continuous-churn', 
    'DATA_SET': '../data/churn_prob_out_35.csv',
    'DATA_TYPES': {'Children': float, 'RatePlan': str},
    'DROP': ['Id', 'CHURN', '3_Class', '5_Class', 'is_test_set'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3, # 'FIXED' for using 'is_test_set'
    'BINARIZER': 'QUANTILE',
    'ALGO': 'GLRM',
    'TARGET_LABEL': 'pChurn',
    'TYPE' : 'CONTINUOUS',
    'EXAMPLE_FEATURE' : 'Est Income',
    'POS_CLASS': None,
    'BASELINE': True
     }
CONFIG3 = { 
    'NAME' : 'bike-demand', 
    'DATA_SET': '../data/SeoulBikeData.csv',
    'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float, 'Humidity': float, 'Visibility (10m)': float, 'RatePlan': str},
    'DROP': ['Date'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'QUANTILE',
    'ALGO': 'GLRM',
    'TARGET_LABEL': 'Rented Bike Count',
    'TYPE' : 'CONTINUOUS',
    'EXAMPLE_FEATURE' : 'Dew point temperature(C)',
    'POS_CLASS': None,
    'BASELINE': True
     }
CONFIG4 = { 
    'NAME' : 'heloc', 
    'DATA_SET': '../data/heloc.csv',
    'DATA_TYPES': {},
    'DROP': ['RiskPerformance'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'QUANTILE',
    'ALGO': 'GLRM',
    'TARGET_LABEL': 'Probabilities',
    'TYPE' : 'CONTINUOUS',
    'EXAMPLE_FEATURE' : 'ExternalRiskEstimate',
    'POS_CLASS': None,
    'BASELINE': True
     }
CONFIG5 = { 
    'NAME' : 'taiwan-credit', 
    'DATA_SET': '../data/TaiwanCreditData.csv',
    'DATA_TYPES': {},
    'DROP': ['DefaultNextMonth'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'QUANTILE',
    'ALGO': 'GLRM',
    'TARGET_LABEL': 'Probabilities',
    'TYPE' : 'CONTINUOUS',
    'EXAMPLE_FEATURE' : 'Amount',
    'POS_CLASS': None,
    'BASELINE': True
     }
CONFIG6 = { 
    'NAME' : 'german-credit-brcg', 
    'DATA_SET': '../data/german_credit_codiert.csv',
    'DATA_TYPES': {},
    'DROP': ['Index'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'QUANTILE',
    'ALGO': 'BRCG',
    'TARGET_LABEL': 'Target',
    'TYPE' : 'BINARY',
    'EXAMPLE_FEATURE' : 'Credit Amount',
    'POS_CLASS': 0,
    'BASELINE': True
    
     }
CONFIG7 = { 
    'NAME' : 'german-credit-ripper', 
    'DATA_SET': '../data/german_credit_codiert.csv',
    'DATA_TYPES': {},
    'DROP': ['Index'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'NATIVE',
    'ALGO': 'RIPPER',
    'TARGET_LABEL': 'Target',
    'TYPE' : 'BINARY',
    'EXAMPLE_FEATURE' : 'Credit Amount',
    'POS_CLASS': 0,
    'BASELINE': True
     }
CONFIG8 = { 
    'NAME' : 'binary-churn-ripper', 
    'DATA_SET': '../data/churn_prob_out_35.csv',
    'DATA_TYPES': {'Children': float, 'RatePlan': str},
    'DROP': ['Id', 'pChurn', '3_Class', '5_Class', 'is_test_set'],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3, # 'FIXED' for using 'is_test_set'
    'BINARIZER': 'QUANTILE',
    'ALGO': 'RIPPER',
    'TARGET_LABEL': 'CHURN',
    'TYPE' : 'BINARY',
    'EXAMPLE_FEATURE' : 'Est Income',
    'POS_CLASS': 'T',
    'BASELINE': True
     }

CONFIG9 = { 
    'NAME' : 'compas-ripper', 
    'DATA_SET': '../data/compas.csv',
    'DATA_TYPES': {},
    'DROP': [],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'QUANTILE',
    'ALGO': 'RIPPER',
    'TARGET_LABEL': 'recidivate-within-two-years',
    'TYPE' : 'BINARY',
    'EXAMPLE_FEATURE' : 'current-charge-degree',
    'POS_CLASS': 0,
    'BASELINE': True
     }

CONFIG10 = { 
    'NAME' : 'mushroom', 
    'DATA_SET': '../data/mushroom.csv',
    'DATA_TYPES': {},
    'DROP': [],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'NATIVE',
    'ALGO': 'RIPPER',
    'TARGET_LABEL': 'Poisonous/Edible',
    'TYPE' : 'BINARY',
    'EXAMPLE_FEATURE' : 'Cap-color',
    'POS_CLASS': "e",
    'BASELINE': True
     }

CONFIG11 = { 
    'NAME' : 'fraud_detection', 
    'DATA_SET': '../data/fraud_detection.csv',
    'DATA_TYPES': {},
    'DROP': [],
    'MODE': 'PREDICTIVE',
    'TRAIN_TEST_SPLIT': 0.3,
    'BINARIZER': 'NATIVE',
    'ALGO': 'RIPPER',
    'TARGET_LABEL': 'Class',
    'TYPE' : 'BINARY',
    'EXAMPLE_FEATURE' : 'V6',
    'POS_CLASS': 1,
    'BASELINE': True
     }





Config_list = [CONFIG1,CONFIG2,CONFIG3,CONFIG4,CONFIG5,CONFIG6,CONFIG7,CONFIG8,CONFIG9,CONFIG10,CONFIG11]

CONFIG = CONFIG11
print('Proceed with configuration:', CONFIG['NAME'])

# %%
import pandas as pd
import numpy as np
import seaborn as sns
# import os
from sklearn.model_selection import train_test_split #, GridSearchCV
from sklearn.metrics import fbeta_score,confusion_matrix,f1_score,precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, r2_score, explained_variance_score, mean_absolute_error, max_error
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import time
import warnings
import re

from aix360.algorithms.rbm import BRCGExplainer, BooleanRuleCG, GLRMExplainer, LinearRuleRegression
if CONFIG['BINARIZER'] == 'QUANTILE':
    from aix360.algorithms.rbm import FeatureBinarizer
elif CONFIG['BINARIZER'] == 'TREE':
    from aix360.algorithms.rbm import FeatureBinarizerFromTrees
if CONFIG['ALGO'] == 'RIPPER':
    from aix360i.algorithms.rule_induction.ripper import Ripper
if CONFIG['ALGO'] == 'CORELS':
    from corels import *





# from explainer import Explainer

# TODO create reference for performance using boosted trees

# import wittgenstein as lw
# from clf_utils import make_tree_dataset, make_forest, score_forest
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# %% [markdown]
# ### Data

# %%
def convert(char):
    if char == CONFIG['POS_CLASS']:
        return 1
    else:
        return 0

df = pd.read_csv(CONFIG['DATA_SET'],dtype=CONFIG['DATA_TYPES'])
df = df.drop(columns=CONFIG['DROP'])
if CONFIG['ALGO'] == 'BRCG':
    df[CONFIG['TARGET_LABEL']] = df[CONFIG['TARGET_LABEL']].map(convert)
    CONFIG['POS_CLASS'] = 1 
    # maybe this could also be achieved through explicit binarization of target vector
df.info()
df.head()

# %%
if CONFIG['TYPE'] == 'BINARY':
    print(df[CONFIG['TARGET_LABEL']].value_counts())
elif CONFIG['TYPE'] == 'CONTINUOUS':
    df[CONFIG['TARGET_LABEL']].describe()
else:
    print('Unrecognized problem type')

# %% [markdown]
# ### Train, Test Split

# %%
if CONFIG['TRAIN_TEST_SPLIT'] == 'FIXED':
    if CONFIG['MODE'] == 'PREDICTIVE':
        train = df[df['is_test_set'] == False]
        test = df[df['is_test_set'] == True]
    elif CONFIG['MODE'] == 'DESCRIPTIVE':
        train = df
        test = df

    train = train.drop(columns=['is_test_set'])
    test = test.drop(columns=['is_test_set'])

    y_train = train[CONFIG['TARGET_LABEL']]
    x_train = train.drop(columns=[CONFIG['TARGET_LABEL']])

    y_test = test[CONFIG['TARGET_LABEL']]
    x_test = test.drop(columns=[CONFIG['TARGET_LABEL']])
else:
    x_train, x_test, y_train, y_test = train_test_split(df.drop(columns=[CONFIG['TARGET_LABEL']]), df[CONFIG['TARGET_LABEL']], test_size=CONFIG['TRAIN_TEST_SPLIT'], random_state=42)

print('Training:', x_train.shape, y_train.shape)
print('Test:', x_test.shape, y_test.shape)

# %% [markdown]
# ### Reference Performance

# %%
if CONFIG['TYPE'] == 'CONTINUOUS':
    print('needs prior encoding of categoricals')
    # gbr = GradientBoostingRegressor(n_estimators=500, random_state=0)
    # gbr.fit(x_train, y_train)
    # # print('Training R^2:', r2_score(yTrain, gbr.predict(dfTrain)))
    # print('Test R^2:', r2_score(y_test, gbr.predict(x_test)))

# %% [markdown]
# ### Binarization

# %%
if CONFIG['BINARIZER'] == 'TREES':
    binarizer =  FeatureBinarizerFromTrees(negations=True, randomState=42) # FeatureBinarizer(negations=False), FeatureBinarizerFromTrees(negations=True, randomState=42)
    binarizer = binarizer.fit(x_train, y_train)
    x_train_bin = binarizer.transform(x_train) #  x_train_bin = binarizer.fit_transform(x_train)
    x_test_bin = binarizer.transform(x_test) #  X_fb = self.fb.fit_transform(X_train)
elif CONFIG['BINARIZER'] == 'QUANTILE':
    binarizer =  FeatureBinarizer(numThresh=9,negations=True) # FeatureBinarizer(negations=False), FeatureBinarizerFromTrees(negations=True, randomState=42)
    binarizer = binarizer.fit(x_train)
    x_train_bin = binarizer.transform(x_train) #  x_train_bin = binarizer.fit_transform(x_train)
    x_test_bin = binarizer.transform(x_test) #  X_fb = self.fb.fit_transform(X_train)  
elif CONFIG['BINARIZER'] == 'NATIVE':
    x_train_bin = x_train
    x_test_bin = x_test
else:
    print('UNRECOGNIZED BINARIZER')

x_train_bin.info() # verbose=True
x_train_bin.head()
x_train_bin[CONFIG['EXAMPLE_FEATURE']][:10]


# %% [markdown]
# ### Rule Induction

# %%
start_time = time.time()
print('Starting training for', CONFIG['ALGO'])

if CONFIG['ALGO'] == 'BRCG':
    estimator = BooleanRuleCG() # Explainer()
    # estimator.train(x_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(x_train_bin, y_train)
elif CONFIG['ALGO'] == 'RIPPER':
    estimator = Ripper()
    estimator.fit(x_train_bin, y_train, pos_value=CONFIG['POS_CLASS'])
elif CONFIG['ALGO'] == 'GLRM':
    linear_model = LinearRuleRegression() # lambda0=0.0005,lambda1=0.0001
    explainer = GLRMExplainer(linear_model)
    explainer.fit(x_train_bin, y_train)
else:
    print('Unrecognized algorithm:', CONFIG['ALGO'])

end_time = time.time()
print('Training time: ' + str(end_time - start_time))

# %% [markdown]
# ### Evaluation

# %%
if CONFIG['TYPE'] == 'BINARY':
    y_pred = estimator.predict(x_test_bin)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Balanced accuracy:', balanced_accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS']))
    print('Recall:', recall_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS']))
    print('F1', f1_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS']))
    print('ConfusionMatrix', confusion_matrix(y_test, y_pred))
    print('F-2', fbeta_score(y_test, y_pred, pos_label=CONFIG['POS_CLASS'], beta= 2))
   
elif CONFIG['TYPE'] == 'CONTINUOUS':
    y_pred = explainer.predict(x_test_bin)
    print(f'R2 Score = {r2_score(y_test, y_pred)}')
    print(f'Explained Variance = {explained_variance_score(y_test, y_pred)}')
    print(f'Mean abs. error = {mean_absolute_error(y_test, y_pred)}')
    print(f'Max error = {max_error(y_test, y_pred)}')
    


# %%
#XGboost for Binary Classification
if CONFIG['TYPE'] == 'BINARY' and CONFIG['BASELINE'] == True:
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    x_train_bin.columns = [regex.sub("_", str(col)) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_train_bin.columns.values]
    x_test_bin.columns = [regex.sub("_", str(col)) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in x_test_bin.columns.values]

    xgb_cl = xgb.XGBClassifier()
    xgb_cl.fit(x_train_bin, y_train)
    preds = xgb_cl.predict(x_test_bin)     
    print('Accuracy:', accuracy_score(y_test, preds))
    print('Balanced accuracy:', balanced_accuracy_score(y_test, preds))
    print('Precision:', precision_score(y_test, preds, pos_label=CONFIG['POS_CLASS']))
    print('Recall:', recall_score(y_test, preds, pos_label=CONFIG['POS_CLASS']))

# %%
if CONFIG['TYPE'] == 'CONTINUOUS':
    explanation = explainer.explain()
    print(explanation)
elif CONFIG['ALGO'] == 'BRCG':
    model = estimator.explain()
    if not model['isCNF']:
        print('Number of rules:', len(model['rules']))
        print(model['rules'])
elif CONFIG['ALGO'] == 'RIPPER':
    print('Rule count: ' + str(sum([len(rules) for rules in estimator.rule_map.values()])))
    print('Rule set:')
    print(estimator.rule_list_to_pretty_string())

# uncomment the following line for a full optimized view of the model as data frame for GLRM rules
# explanation.style

# %% [markdown]
# # Analysis of CSV

# %%
from pathlib import Path
import matplotlib.pyplot as plt

# create df and label target label new
df_list = []
csv_list = []
for i in range(len(Config_list)):
    
    if Config_list[i]['TYPE'] == "BINARY":
        if Config_list[i]["DATA_SET"] not in csv_list:
            temp_df = pd.read_csv(Config_list[i]["DATA_SET"])
            csv_list.append(Config_list[i]["DATA_SET"])
            temp_df = temp_df.drop(columns=Config_list[i]['DROP'])
            temp_df= temp_df.rename(columns={temp_df[Config_list[i]['TARGET_LABEL']].name : 'TARGET_LABEL'})
            df_list.append(temp_df)
        
eval_df = pd.DataFrame(csv_list, columns=['Data_Set'])
eval_df["Target_1_pos"] = pd.Series()
eval_df["Target_2_neg"] = pd.Series()
eval_df["IB_Ratio"] = pd.Series()
eval_df["Size"] = pd.Series()

eval_df


# %%
# calculate imbalance Ratio

for frame in range(len(df_list)):
    metric = df_list[frame]["TARGET_LABEL"].value_counts()
    if metric[0] > metric[1]:
        eval_df["Target_1_pos"].iloc[frame] = metric[1]
        eval_df["Target_2_neg"].iloc[frame] = metric[0]
    else:
        eval_df["Target_1_pos"].iloc[frame] = metric[0]
        eval_df["Target_2_neg"].iloc[frame] = metric[1]
    df_size = len(df_list[frame])
    eval_df["IB_Ratio"].iloc[frame] = eval_df["Target_1_pos"].iloc[frame]/eval_df["Target_2_neg"].iloc[frame]
    eval_df["Size"].iloc[frame]    = df_size
    print(metric)
    


# %%
eval_df

# %%



