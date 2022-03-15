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
'POS_CLASS': 1,
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
'BINARIZER': 'NATIVE',
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
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float,'Humidity(%)': float,'Visibility (10m)': float},
'DROP': ['Rented Bike Count', 'Unnamed: 0'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Target',
'POS_CLASS': 1,
'ONEHOT': False,
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
},

"CONFIG16" : { 
'NAME' : 'binary-churn-r2n', 
'DATA_SET': '../data/churn_prob_out_35.csv',
'DATA_TYPES': {'Children': float, 'RatePlan': str},
'DROP': ['Id', 'pChurn', '3_Class', '5_Class', 'is_test_set'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3, # 'FIXED' for using 'is_test_set'
'BINARIZER': 'NATIVE',
'ALGO': 'R2N',
'TARGET_LABEL': 'CHURN',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Est Income',
'POS_CLASS': 'T'
}

}

# Imbalanced Configs

config_dict_imbalanced= {"CONFIG-I1" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {'Duration in Month': float, 'Credit Amount': float, 'Installmentrate %': float, 'PresentResidence': float, 'Age in years': float, 'Number existing Credits': float, 'Number people liable': float},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0,
'BASELINE': False,
'ONEHOT': False


},
"CONFIG-I2" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0,
'BASELINE': False,
'ONEHOT': False


},
"CONFIG-I3" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0,
'BASELINE': False,
'ONEHOT': False


},
"CONFIG-I4" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0,
'BASELINE': False,
'ONEHOT': False


},

"CONFIG-I5" : { 
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
'BASELINE': False,
'ONEHOT': False


},

"CONFIG-I6" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0,
'BASELINE': False,
'ONEHOT': False


},

"CONFIG-I7" : { 
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
'BASELINE': False,
'ONEHOT': False


},


"CONFIG-I8" : { 
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
'ONEHOT': False
},
"CONFIG-I9" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I10" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I11" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I12" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I13" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},


"CONFIG-I14" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},


"CONFIG-I15" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},
"CONFIG-I16" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I17" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I18" : { 
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
'BASELINE': False,
'ONEHOT': False
},


"CONFIG-I19" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'BRCG',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I20" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},

"CONFIG-I21" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': False,
'ONEHOT': False
},


"CONFIG-I22" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float, 'yearlyReimbursement': float},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I23" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'ONEHOT': False
},


"CONFIG-I24" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'ONEHOT': False
}
,


"CONFIG-I25" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'ONEHOT': False
},


"CONFIG-I26" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'BRCG',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'ONEHOT': False
}

,


"CONFIG-I27" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'ONEHOT': False
},
"CONFIG-I28" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'ONEHOT': False
},
"CONFIG-I29" : { 
'NAME' : 'german-credit-brcg', 
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
'ONEHOT': True
},


"CONFIG-I30" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {'Duration in Month': float, 'Credit Amount': float, 'Installmentrate %': float, 'PresentResidence': float, 'Age in years': float, 'Number existing Credits': float, 'Number people liable': float},
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
'UNDERSAMP': True,
'ONEHOT': False
},

"CONFIG-I31" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'BASELINE': True,
'ONEHOT': True
},
"CONFIG-I32" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': True,
'ONEHOT': False
},


"CONFIG-I33" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {'Duration in Month': float, 'Credit Amount': float, 'Installmentrate %': float, 'PresentResidence': float, 'Age in years': float, 'Number existing Credits': float, 'Number people liable': float},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
},


"CONFIG-I34" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
},


"CONFIG-I35" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
},


"CONFIG-I36" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
},


"CONFIG-I37" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
},


"CONFIG-I38" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
}

,


"CONFIG-I39" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
},


"CONFIG-I40" : { 
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
'UNDERSAMP': True,
'ONEHOT': False
},


"CONFIG-I41" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float,'Humidity(%)': float,'Visibility (10m)': float},
'DROP': ["Rented Bike Count", "Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1,
'BASELINE': True,
'UNDERSAMP': False,
'ONEHOT': True
},


"CONFIG-I42" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {},
'DROP': ["Rented Bike Count", "Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I43" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {},
'DROP': ["Rented Bike Count", "Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I44" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {},
'DROP': ["Rented Bike Count", "Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
}
,


"CONFIG-I45" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {},
'DROP': ["Rented Bike Count", "Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I46" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {},
'DROP': ["Rented Bike Count", "Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I47" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {},
'DROP': ["Rented Bike Count", "Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I48" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {'WeekOfMonth': float, 'WeekOfMonthClaimed': float,'Age': float,'PolicyNumber': float,'Age': float,'RepNumber': float,'Deductible': float,'DriverRating': float,'Year': float},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': True

},


"CONFIG-I49" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I50" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I51" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I52" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'BRCG',
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I53" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I54" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1,
'BASELINE': False,
'UNDERSAMP': False,
'ONEHOT': False
},


"CONFIG-I55" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float, 'yearlyReimbursement': float},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': False
}
,


"CONFIG-I56" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'BRCG',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': True
},


"CONFIG-I57" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'BRCG',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': True
},


"CONFIG-I58" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'CORELS',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': True
}

,


"CONFIG-I59" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'CORELS',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': True
}
,


"CONFIG-I60" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': True
}
,


"CONFIG-I61" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False,
'BASELINE': False,
'UNDERSAMP': True,
'ONEHOT': True
},

"CONFIG-I62" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection_duenn.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'TREES',
'ALGO': 'RIPPER',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1

},
"CONFIG-I63" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'QUANTILE',
'ALGO': 'R2N',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0
},
"CONFIG-I64" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'R2N',
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1
},
"CONFIG-I65" : { 
'NAME' : 'binary_bike', 
'DATA_SET': '../data/binary_bike.csv',
'DATA_TYPES': {},
'DROP': ['Rented Bike Count', 'Unnamed: 0'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'BINARIZER': 'NATIVE',
'ALGO': 'R2N',
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Target',
'POS_CLASS': 1

}


}




Config_list = []

for config in config_dict_imbalanced.items():
    Config_list.append(config)
    



