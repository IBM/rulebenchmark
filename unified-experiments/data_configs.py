
# add datasets here

CONFIG_DICT_IMBALANCED = {


   

"CONFIG-I1" : { 
'NAME' : 'mushroom', 
'DATA_SET': '../data/mushroom.csv',
'DATA_TYPES': {},
'DROP': [],
'TARGET_LABEL': 'Poisonous/Edible',
'EXAMPLE_FEATURE' : 'Cap-color',
'POS_CLASS': "e",
'META_DATA': {'use_case': "mush_room",'flag': "organic"}
},


"CONFIG-I2" : { 
 'NAME' : 'bike_imbalanced', 
 'DATA_SET': '../data/binary_bike_imbalanced.csv',
 'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float,'Humidity_percent': float,'Visibility_10m': float},
 'DROP': ["Unnamed: 0","Rented Bike Count"],
 'TARGET_LABEL': 'Target',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'Hour',
 'POS_CLASS': 1,
'META_DATA': {'use_case': "rent_bike",'flag': "organic"}
 
 },


 "CONFIG-I3" : { 
 'NAME' : 'bike_imbalanced', 
 'DATA_SET': '../data/binary_bike.csv',
 'DATA_TYPES': {'Rented Bike Count': float, 'Hour': float,'Humidity_percent': float,'Visibility_10m': float},
 'DROP': ["Unnamed: 0","Rented Bike Count"],
 'TARGET_LABEL': 'Target',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'Hour',
 'POS_CLASS': 1,
'META_DATA': {'use_case': "rent_bike",'flag': "organic"}
 
 },

"CONFIG-I4" : { 
 'NAME' : 'fraud_detection', 
 'DATA_SET': '../data/fraud_detection.csv',
 'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
 'DROP': [],
 'TARGET_LABEL': 'Class',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'V6',
 'POS_CLASS': 1,
'META_DATA': {'use_case': "fraud_detection",'flag': "organic"}
 },

"CONFIG-I5" : { 
'NAME' : 'binary-churn', 
'DATA_SET': '../data/churn_prob_out_35.csv',
'DATA_TYPES': {'Children': float, 'RatePlan': str},
'DROP': ['Id', 'pChurn', '3_Class', '5_Class', 'is_test_set'],
'TARGET_LABEL': 'CHURN',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Est Income',
'POS_CLASS': 'T',
'META_DATA': {'use_case': "churn",'flag': "organic"}
},

"CONFIG-I6" : { 
'NAME' : 'german-credit', 
'DATA_SET': '../data/german_credit_codiert.csv',
'DATA_TYPES': {'Duration in Month': float, 'Credit Amount': float, 'Installmentrate': float, 'PresentResidence': float, 'Age in years': float, 'Number existing Credits': float, 'Number people liable': float},
'DROP': ['Index'],
'TARGET_LABEL': 'Target',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'Credit Amount',
'POS_CLASS': 0,
'META_DATA': {'use_case': "credit_approval",'flag': "organic"}
},

"CONFIG-I7" : { 
'NAME' : 'taiwan_binary', 
'DATA_SET': '../data/TaiwanCreditData.csv',
'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
'DROP': ["Probabilities"],
'TARGET_LABEL': 'DefaultNextMonth',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'PayAmount_Apr',
'POS_CLASS': 1,
'META_DATA': {'use_case': "credit_approval",'flag': "organic"}
},


"CONFIG-I8" : { 
'NAME' : 'compas', 
'DATA_SET': '../data/compas.csv',
'DATA_TYPES': {},
'DROP': ["Unnamed: 0"],
'TARGET_LABEL': 'recidivate-within-two-years',
'TYPE' : 'BINARY',
'EXAMPLE_FEATURE' : 'current-charge-degree',
'POS_CLASS': 0,
'META_DATA': {'use_case': "recidivism risk",'flag': "organic"}
},


 "CONFIG-I9" : { 
 'NAME' : 'fraud_detection', 
 'DATA_SET': '../data/fraud_detection.csv',
 'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
 'DROP': [],
 'TARGET_LABEL': 'Class',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'V6',
 'POS_CLASS': 1,
'META_DATA': {'use_case': "fraud_detection",'flag': "organic"}
 },



 "CONFIG-I10" : { 
 'NAME' : 'miniloan', 
 'DATA_SET': '../data/miniloan-decisions-100K.csv',
 'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float, 'yearlyReimbursement': float},
 'DROP': ["Unnamed: 0", "name"],
 'TARGET_LABEL': 'approval',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'income',
 'POS_CLASS': False,
'META_DATA': {'use_case': "loan_approval",'flag': "synthetic"}
 },

 "CONFIG-I11" : { 
 'NAME' : 'fraud_detection_ib', 
 'DATA_SET': '../data/fraud_detection_duenn.csv',
 'DATA_TYPES': {'Amount': float, 'Age': float, 'Bill_Sep': float, 'Bill_Aug': float, 'Bill_Jul': float, 'Bill_Jun': float, 'Bill_May': float, 'Bill_Apr': float, 'PayAmount_Sep': float, 'PayAmount_Aug': float, 'PayAmount_Jul': float, 'PayAmount_Jun': float, 'PayAmount_May': float, 'PayAmount_Apr': float},
 'DROP': ["Unnamed: 0"],
 'TARGET_LABEL': 'Class',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'V6',
 'POS_CLASS': 1,
'META_DATA': {'use_case': "fraud_detection",'flag': "organic"}
 },



 "CONFIG-I12" : { 
 'NAME' : 'fraud_oracle', 
 'DATA_SET': '../data/fraud_oracle_clean.csv',
 'DATA_TYPES': {'WeekOfMonth': float, 'WeekOfMonthClaimed': float,'Age': float,'PolicyNumber': float,'Age': float,'RepNumber': float,'Deductible': float,'DriverRating': float,'Year': float},
 'DROP': ["Unnamed: 0"],
 'TARGET_LABEL': 'FraudFound_P',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'Sex',
 'POS_CLASS': 1,
'META_DATA': {'use_case': "insurance_fraud_detection",'flag': "organic"}
 },

 "CONFIG-I13" : { 
 'NAME' : 'miniloan_ib', 
 'DATA_SET': '../data/miniloan_duenn.csv',
 'DATA_TYPES': {'creditScore': float, 'income': float, 'loanAmount': float, 'monthDuration': float, 'yearlyReimbursement': float},
 'DROP': ["Unnamed: 0"],
 'TARGET_LABEL': 'approval',
 'TYPE' : 'BINARY',
 'EXAMPLE_FEATURE' : 'income',
 'POS_CLASS': False,
'META_DATA': {'use_case': "loan_approval",'flag': "synthetic"}
},

"CONFIG-I14" : { 
'NAME' : 'heloc', 
'DATA_SET': '../data/heloc.csv',
'DATA_TYPES': {'ExternalRiskEstimate': float, 'MSinceOldestTradeOpen': float, 'MSinceMostRecentTradeOpen': float, 'AverageMInFile': float, 'NumSatisfactoryTrades': float,'NumTrades60Ever2DerogPubRec': float, 'NumTrades90Ever2DerogPubRec': float, 'PercentTradesNeverDelq': float, 'MSinceMostRecentDelq': float, 'MaxDelq2PublicRecLast12M': float,'MaxDelqEver': float, 'NumTotalTrades': float, 'NumTradesOpeninLast12M': float, 'PercentInstallTrades': float, 'MSinceMostRecentInqexcl7days': float,'NumInqLast6M': float, 'NumInqLast6Mexcl7days': float, 'NetFractionRevolvingBurden': float, 'NetFractionInstallBurden': float, 'NumRevolvingTradesWBalance': float,'NumInstallTradesWBalance': float, 'NumBank2NatlTradesWHighUtilization': float, 'PercentTradesWBalance': float},
'DROP': ['Probabilities'],
'TARGET_LABEL': 'RiskPerformance',
'EXAMPLE_FEATURE' : 'ExternalRiskEstimate',
'POS_CLASS': "Good",
'META_DATA': {'use_case': "credit_risk",'flag': "organic"}

},



}