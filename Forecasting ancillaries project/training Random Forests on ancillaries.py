from IPython.core.display import Markdown
import pandas as pd
import numpy as np
import datetime as dt
import time
import pickle
import os
cwd=os.getcwd()
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss,f1_score, recall_score
import seaborn as sb
import matplotlib.pyplot as plt
from copy import deepcopy


customer_data = pd.read_csv("customer_data.csv")


# Getting rid of unnecessary columns such as the ones pinpointed in the powerpoint and also 
customer_data = customer_data.drop(columns = ["UPGRADE_SALE_DT", "BAGGAGE_SALE_DT", "SEATING_SALE_DT", "TVL_CBN_CD", "BKG_ORDER_ID", "PARTY_ID", "Unnamed: 0"])

# Stripping whitespaces from each column in the dataframe
def strip_df(df):
    '''
    Given a df, return a copy of the dataframe
    with whitespace removed from the front and back 
    of each entry.
    '''
    df_copy = deepcopy(df)
    
    for col_name in list(df.columns):
        if df_copy[col_name].dtype == 'object':
            df_copy[col_name] = df_copy[col_name].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    return df_copy
customer_data = strip_df(customer_data)


# getting rid of invalid data
customer_data.drop(customer_data.index[customer_data['BKD_CBN_CD'] == '  '], inplace=True)
customer_data.dropna(inplace = True)


# Setting features to categorical
categorical_features = ["FURTHEST_STN_CD", "ROUTE_GROUP", "BUS_LEIS_IND", "jny_typ", "JNY_CAT", "POS", "MKTG_AREA", "MKTG_REGION", "BKG_CHANNEL", "EC_TIER", "BKD_CBN_CD", "BAH_INDICATOR"]
customer_data[categorical_features] = customer_data[categorical_features].astype("category")

# Setting features to numerical datetype
datetype_features = ["BKG_DT", "FIRST_FLT_DT", "FINAL_FLT_DT"]
for column in datetype_features:
    customer_data[column] = pd.to_datetime(customer_data[column], format='%m/%d/%Y')

# Performing one hot encoding 
customer_data['LONG_HAUL_IND'] = customer_data['LONG_HAUL_IND'].replace(['Y','N'],[1,0])
customer_data['SATURDAY_NIGHT_STAY_IND'] = customer_data['SATURDAY_NIGHT_STAY_IND'].replace(['Y','N'],[1,0])
customer_data['REDEMPTION_BOOKING_IND'] = customer_data['REDEMPTION_BOOKING_IND'].replace(['Y','N'],[1,0])


# Function to standardise a column
def min_max_scaling(column):
    return (column - column.min()) / (column.max() - column.min())

customer_data["L36M_REVENUE"] = min_max_scaling(customer_data["L36M_REVENUE"])


# One hot encoding data using dummies
customer_data = pd.get_dummies(customer_data, columns=['ROUTE_GROUP','BUS_LEIS_IND','EC_TIER','BKD_CBN_CD','MKTG_REGION','BKG_CHANNEL', 'BAH_INDICATOR'])

# Getting rid of these features temporarily
customer_data = customer_data.drop(columns=["FURTHEST_STN_CD", "jny_typ", "JNY_CAT", "POS", "MKTG_AREA", "HBO_IND", "BKG_DT", "FIRST_FLT_DT", "FINAL_FLT_DT", ])


# Grouping the Gold tiers together
customer_data['EC_TIER_Gold'] = customer_data[['EC_TIER_Gold', 'EC_TIER_Gold For Life', 'EC_TIER_Gold Guest List', 'EC_TIER_Gold Guest List For Life']].any(axis=1)
customer_data.drop(columns=['EC_TIER_Gold For Life', 'EC_TIER_Gold Guest List', 'EC_TIER_Gold Guest List For Life'], inplace=True)


x = customer_data.drop(columns=["BAGGAGE_TARGET", "P4S_TARGET", "UPG_TARGET"])
y = customer_data[["BAGGAGE_TARGET", "P4S_TARGET", "UPG_TARGET"]]

# Stratify parameter just ensures that we get the same proportion of each of the y-data in the train and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, stratify=y)


imbalance_baggage = len(customer_data[(customer_data['BAGGAGE_TARGET'] == 1)])/len(customer_data)
imbalance_p4s = len(customer_data[(customer_data['P4S_TARGET'] == 1)])/len(customer_data)
imbalance_upgrade = len(customer_data[(customer_data['UPG_TARGET'] == 1)])/len(customer_data)
print('Baggage:',imbalance_baggage)
print('Pay for seating:',imbalance_p4s)
print('Upgrade:',imbalance_upgrade)



def f1(y_true,y_pred):
    f1=f1_score(y_true, y_pred)
    print(f'F1 Score:{f1}')
    return f1

def recall(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    print(f'Recall Score: {recall}')
    return recall

def f1_and_recall(y_true, y_pred):
    f1=f1_score(y_true, y_pred)
    print(f'F1 Score:{f1}')
    recall = recall_score(y_true, y_pred)
    print(f'Recall Score: {recall}')
    return f1, recall

"""

Random forests paid for baggage

"""



features = x_train.columns
features_baggage = features
run_features_RF_baggage = []
run_scores_RF_baggage = []
for r in range(len(features)):
    model_baggage = RandomForestClassifier()
    trained_model_baggage = model_baggage.fit(x_train[features_baggage], y_train['BAGGAGE_TARGET'])
    RF_model_predicted_baggage = trained_model_baggage.predict(x_test[features_baggage], )
    baggage_score = f1_and_recall(y_true=y_test['BAGGAGE_TARGET'],y_pred=RF_model_predicted_baggage)
    run_features_RF_baggage.append(features_baggage)
    run_scores_RF_baggage.append(baggage_score)
    feature_importances = trained_model_baggage.feature_importances_
    indices = np.argsort(feature_importances)
    new_features = np.array(features_baggage)[indices][1:]
    features_baggage = new_features
print('Model trained!')

liss = [run_features_RF_baggage, run_scores_RF_baggage]
pickle.dump(liss, open("most_important_features_RF_baggage", 'wb'))



"""

Random forests paid for seat

"""
features = x_train.columns
features_p4s = features
run_features_RF_p4s = []
run_scores_RF_p4s = []
for r in range(len(features)):
    model_p4s = RandomForestClassifier()
    trained_model_p4s = model_p4s.fit(x_train[features_p4s], y_train['P4S_TARGET'])
    RF_model_predicted_p4s = trained_model_p4s.predict(x_test[features_p4s], )
    p4s_score = f1_and_recall(y_true=y_test['P4S_TARGET'],y_pred=RF_model_predicted_p4s)
    run_features_RF_p4s.append(features_p4s)
    run_scores_RF_p4s.append(p4s_score)
    feature_importances = trained_model_p4s.feature_importances_
    indices = np.argsort(feature_importances)
    new_features = np.array(features_p4s)[indices][1:]
    features_p4s = new_features
print('Model trained!')

liss = [run_features_RF_p4s, run_scores_RF_p4s]
pickle.dump(liss, open("most_important_features_RF_p4s", 'wb'))


"""

Random forests paid for upgrade

"""
features = x_train.columns
features_upgrade = features
run_features_RF_upgrade = []
run_scores_RF_upgrade = []
for r in range(len(features)):
    model_upgrade = RandomForestClassifier()
    trained_model_upgrade = model_upgrade.fit(x_train[features_upgrade], y_train['UPG_TARGET'])
    RF_model_predicted_upgrade = trained_model_upgrade.predict(x_test[features_upgrade], )
    upgrade_score = f1_and_recall(y_true=y_test['UPG_TARGET'],y_pred=RF_model_predicted_upgrade)
    run_features_RF_upgrade.append(features_upgrade)
    run_scores_RF_upgrade.append(upgrade_score)
    feature_importances = trained_model_upgrade.feature_importances_
    indices = np.argsort(feature_importances)
    new_features = np.array(features_upgrade)[indices][1:]
    features_upgrade = new_features
print('Model trained!')

liss = [run_features_RF_upgrade, run_scores_RF_upgrade]
pickle.dump(liss, open("most_important_features_RF_upgrade", 'wb'))

