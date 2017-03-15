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

chunks = 5
df = pd.DataFrame()
for n in range(1, chunks+1):
    log.info('Loading dataframe...#{}'.format(n))
    df = df.append(pd.read_hdf(commons.FILE_DF + "." + str(n), key='santander'))

df = df[~df['is_test_data']]
df.drop(['is_test_data'], inplace=True, axis=1)
df.drop(commons.indicators_ignored, inplace=True, axis=1)

df = add_activations(df)
df.drop(commons.indicators ,inplace=True,axis=1)

#Drop excessive months. Keep 2015-06, 2016-03/05.

log.info("Before: {}".format(df.shape))
df = df[df['fecha_dato'].isin([pd.Timestamp('2015-06'), pd.Timestamp('2016-03'), pd.Timestamp('2016-04'), pd.Timestamp('2016-05')])]
log.info("After: {}".format(df.shape))
#df.drop(['ncodpers', 'fecha_dato'], inplace=True, axis=1)
df.drop(['ncodpers'], inplace=True, axis=1)
df['fecha_dato'] = df.apply(lambda r: r['fecha_dato'].month, axis=1)

activation_columns=["act_" + i for i in commons.indicators]

for ind in commons.indicators:
    log.info("Building model for {}".format(ind))
    X = df.drop(activation_columns, axis = 1)
    Y = df["act_" + ind]
#    log.info("X: {}, Y: {}".format(list(X.columns.values), Y.name))
    clf = xgb.XGBClassifier(objective = 'binary:logistic', nthread=8, silent=1, max_depth=6)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    log.info("Feature importances: {}".format(sorted(zip(list(X), clf.feature_importances_), key=lambda x: x[1])))
    log.info("Accuracy: {:.2%}, precision: {:.2%}, recall: {:.2%}".format(
        accuracy_score(Y_test, Y_pred),
        precision_score(Y_test, Y_pred),
        recall_score(Y_test, Y_pred)))
