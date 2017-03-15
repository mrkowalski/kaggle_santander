
import commons, sys, os
import logging as log
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def add_activations(df):
    for ind in commons.indicators:
        log.info("Adding activations for {}".format(ind))
        ind_prev = ind + "_1"
        res = df[ind].sub(df[ind_prev])
        res[res < 0] = 0
        df["act_" + ind] = res.fillna(0)
    return df

df = df[~df['is_test_data']]
df.drop(['is_test_data'], inplace=True, axis=1)
df.drop(commons.indicators_ignored, inplace=True, axis=1)

df = add_activations(df)
df.drop(commons.indicators ,inplace=True,axis=1)

#Drop excessive months. Keep 2015-06, 2016-03/05.

log.info("Before: {}".format(df.shape))
df = df[~df['fecha_dato'].isin(pd.Period('2015-06'), pd.Period('2016-03'), pd.Period('2016-04'), pd.Period('2016-05'))]
log.info("After: {}".format(df.shape))
#df.drop(['ncodpers', 'fecha_dato'], inplace=True, axis=1)
df.drop(['ncodpers'], inplace=True, axis=1)

activation_columns=["act_" + i for i in commons.indicators]

for ind in commons.indicators:
    log.info("Building model for ".format(ind))
