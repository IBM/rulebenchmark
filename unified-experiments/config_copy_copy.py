
# add datasets here


config_dict_imbalanced= {"CONFIG-I1" : { 
'NAME' : 'german-credit-brcg', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {'Duration in Month': float, 'Credit Amount': float, 'Installmentrate %': float, 'PresentResidence': float, 'Age in years': float, 'Number existing Credits': float, 'Number people liable': float},
'DROP': ['Index'],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0



},


"CONFIG-I2" : { 
'NAME' : 'fraud_detection', 
'DATA_SET': '../data/fraud_detection.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': [],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1

},





"CONFIG-I3" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': ["Probabilities"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1
},


"CONFIG-I4" : { 
'NAME' : 'miniloan', 
'DATA_SET': '../data/miniloan-decisions-100K.csv',
'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float, 'yearlyReimbursement': float},
'DROP': ["Unnamed: 0", "name"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False

},

"CONFIG-I5" : { 
'NAME' : 'fraud_detection_ib', 
'DATA_SET': '../data/fraud_detection_duenn.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'Class',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'V6',
'POS_CLASS': 1

},


"CONFIG-I6" : { 
'NAME' : 'bike_imbalanced', 
'DATA_SET': '../data/binary_bike_imbalanced.csv',
'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float,'Humidity(%)': float,'Visibility (10m)': float},
'DROP': ["Unnamed: 0","Rented Bike Count"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Hour',
'POS_CLASS': 1

},


"CONFIG-I7" : { 
'NAME' : 'fraud_oracle', 
'DATA_SET': '../data/fraud_oracle_clean.csv',
'DATA_TYPES': {'WeekOfMonth': float, 'WeekOfMonthClaimed': float,'Age': float,'PolicyNumber': float,'Age': float,'RepNumber': float,'Deductible': float,'DriverRating': float,'Year': float},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'FraudFound_P',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Sex',
'POS_CLASS': 1

},

"CONFIG-I8" : { 
'NAME' : 'miniloan_ib', 
'DATA_SET': '../data/miniloan_duenn.csv',
'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float, 'yearlyReimbursement': float},
'DROP': ["Unnamed: 0"],
'MODE': 'PREDICTIVE',
'TRAIN_TEST_SPLIT': 0.3,
'TARGET_LABEL': 'approval',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'income',
'POS_CLASS': False
}

}



Config_list = []

for config in config_dict_imbalanced.items():
    Config_list.append(config)
    
