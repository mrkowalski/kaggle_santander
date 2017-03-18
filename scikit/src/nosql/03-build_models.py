import commons, sys, os
import logging as log
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def save_model(clf, feature):
    if not os.path.exists("models"): os.makedirs("models")
    joblib.dump(clf, "models/" + feature + ".pkl")

def add_activations(df):
    for ind in commons.indicators:
        log.info("Adding activations for {}".format(ind))
        ind_prev = ind + "_1"
        res = df[ind].sub(df[ind_prev])
        res[res < 0] = 0
        df["act_" + ind] = res.fillna(0)
    return df

chunks = 6
df = pd.DataFrame()
for n in range(1, chunks+1):
    log.info('Loading dataframe...#{}'.format(n))
    df = df.append(pd.read_hdf(commons.FILE_DF + "." + str(n), key='santander'))

df = add_activations(df)
df.drop(commons.indicators ,inplace=True,axis=1)
df.drop(['ncodpers'], inplace=True, axis=1)

activation_columns=["act_" + i for i in commons.indicators]

for ind in commons.indicators:
    log.info("Building model for {}".format(ind))
    X = df.drop(activation_columns, axis = 1)
    Y = df["act_" + ind]
    if sum(Y) > 0:
        ratio = (Y.shape[0] - sum(Y)) / sum(Y)
        log.info("Number of positive cases: {}".format(sum(Y)))
        log.info("Negative / positive: {}".format(ratio))
        clf = xgb.XGBClassifier(objective = 'binary:logistic', nthread=8, silent=1, max_depth=6, scale_pos_weight=ratio)
        clf = clf.fit(X, Y)
        log.info("Feature importances: {}".format(sorted(zip(list(X), clf.feature_importances_), key=lambda x: x[1])))
        save_model(clf, ind)
    else:
        log.info("Skipping {} due to no positive cases.".format(ind))
