import pandas as pd
from sklearn.model_selection import train_test_split 
import time


from aix360i.algorithms.rule_induction.r2n.r2n_algo import R2Nalgo

CONFIG = { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': ["Probabilities"],
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'META_DATA': {'use_case': "credit_approval",'flag': "organic"}
}

# For some rule induction algorithms the pos class is always value 1
def convert(char):
    if char == CONFIG['POS_CLASS']:
        return 1
    else:
        return 0


TRAIN_TEST_SPLIT =  0.3

script_start_time = time.time()
print('Job started at {}.'.format(time.ctime()))

df = pd.read_csv(CONFIG['DATA_SET'],dtype=CONFIG['DATA_TYPES'])
print('Read', len(df), 'rows from', CONFIG['DATA_SET'])
df = df.drop(columns=CONFIG['DROP'])

df[CONFIG['TARGET_LABEL']] = df[CONFIG['TARGET_LABEL']].map(convert)
POS_CLASS = 1

x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns=[CONFIG['TARGET_LABEL']]), 
    df[CONFIG['TARGET_LABEL']], 
    test_size=TRAIN_TEST_SPLIT, 
    random_state=42)

x_train_bin = x_train
x_test_bin = x_test

estimator = R2Nalgo(n_seeds=2, max_epochs=5*10**2, min_temp = 10**-4, decay_rate=0.98, coef = 5*10**-4, normalize_num=True,negation=False)
estimator.fit(x_train_bin, y_train)

y_pred = estimator.predict(x_test_bin)

rule_set = estimator.export_rules_to_trxf_dnf_ruleset()
conjunctions = rule_set.list_conjunctions()
nof_rules = len(conjunctions)
conjunction_len = [conjunction.len() for conjunction in conjunctions]
preds_sum = sum(conjunction_len)
preds_max = max(conjunction_len, default=0)

print(nof_rules, preds_sum, preds_max)

script_end_time = time.time()
script_time = script_end_time - script_start_time
print('Job finished at {} in {} seconds.'.format(time.ctime(), script_time))
