config_dict= {"CONFIG1" : { 
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
'BASELINE': True,
'USECASE': None
},
"CONFIG2" : { 
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
'BASELINE': True,
'USECASE': None
},
"CONFIG3" : { 
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
'BASELINE': True,
'USECASE': None
},
"CONFIG4" : { 
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
'BASELINE': True,
'USECASE': None
},
"CONFIG5" : { 
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
'BASELINE': True,
'USECASE': None
},
"CONFIG6" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0,
'BASELINE': True,
'USECASE': None

},
"CONFIG7" : { 
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
'BASELINE': True,
'USECASE': None
},
"CONFIG8" : { 
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
'BASELINE': True,
'USECASE': None
},

"CONFIG9" : { 
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
'BASELINE': True,
'USECASE': None
},

"CONFIG10" : { 
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
'BASELINE': True,
'USECASE': None
},

"CONFIG11" : { 
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
'BASELINE': True,
'USECASE': None
},
"CONFIG12" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': True,
'USECASE': None
},
"CONFIG13" : { 
'NAME' : 'heloc', 
'DATA_SET': '../data/heloc.csv',
'DATA_TYPES': {},
'DROP': ['Probabilities'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'RiskPerformance',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'ExternalRiskEstimate',
'POS_CLASS': 'Good',
'BASELINE': True,
'USECASE': None
},"CONFIG14" : { 
'NAME' : 'binary_bike', 
'DATA_SET': '../data/binary_bike.csv',
'DATA_TYPES': {},
'DROP': ['Rented Bike Count', 'Unnamed: 0'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Target',
'POS_CLASS': 1,
'BASELINE': True,
'USECASE': None
},"CONFIG15" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': True,
'USECASE': None
}
}




Config_list = []

for config in config_dict.items():
    Config_list.append(config)
    



