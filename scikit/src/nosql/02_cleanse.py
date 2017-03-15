#BUG!!!! WHAT ABOUT FIRST MONTH? IT SHOULD MOST LIKELY CONTAIN NO ACTIVATION FLAGS.
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

def show_activation_stats(df):
    log.info("Activation stats:")
    df = df[~df['is_test_data']]
    log.info("All cases: {}".format(df.shape[0]))
    for ind in commons.indicators:
        log.info("{}: {}. Distinct={}".format(ind, sum(df["act_" + ind] == 1), df["act_" + ind].unique()))

chunks = 1
df = pd.DataFrame()
for n in range(1, chunks+1):
    log.info('Loading dataframe...#{}'.format(n))
    df = df.append(pd.read_hdf(commons.FILE_DF + "." + str(n), key='santander'))

df = df[~df['is_test_data']]
df.drop(['is_test_data'], inplace=True, axis=1)
df.drop(commons.indicators_ignored, inplace=True, axis=1)

df = add_activations(df)
df.drop(commons.indicators ,inplace=True,axis=1)
#df.drop([i + "_1" for i in (commons.indicators + commons.indicators_ignored)] ,inplace=True,axis=1)
df.drop(['ncodpers', 'fecha_dato'], inplace=True, axis=1)

activation_columns=["act_" + i for i in commons.indicators]

log.info('Adding any_activation column...')

df_with_activations    = df[df[activation_columns].sum(axis=1) != 0].copy()
df_without_activations = df[df[activation_columns].sum(axis=1) == 0].copy()

del df

df_with_activations['any_activation']    = 1
df_without_activations['any_activation'] = 0

log.info("Training activation classifier...")
df_any_activation = df_with_activations.append(df_without_activations).drop(activation_columns, axis=1)
X = df_any_activation.drop(['any_activation'], axis=1)
Y = df_any_activation['any_activation']
log.info("X: {}, Y: {}".format(list(X.columns.values), Y.name))
clf = xgb.XGBClassifier(objective = 'binary:logistic', nthread=8, silent=1, max_depth=6)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
log.info("Accuracy: {:.2%}, precision: {:.2%}, recall: {:.2%}".format(
    accuracy_score(Y_test, Y_pred),
    precision_score(Y_test, Y_pred),
    recall_score(Y_test, Y_pred)))

#save_model(clf.fit(X, Y), 'activation')

log.info("Preparing products classifier data...")
df_with_activations['activation'] = df_with_activations.apply(lambda r: "".join([str(int(e)) for e in r[activation_columns]]), axis=1)
df_with_activations['activation'] = df_with_activations['activation'].astype('category')
X = df_with_activations.drop(activation_columns + ['activation', 'any_activation'], axis=1)
Y = df_with_activations['activation']#.cat.codes
log.info("X: {}, Y: {}".format(list(X.columns.values), Y.name))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)

log.info("Training products classifier...")
clf = xgb.XGBClassifier(objective = 'multi:softmax', nthread=8, silent=1, max_depth=6)#, num_class=len(df_with_activations['activation'].cat.categories))
clf = clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
log.info("Accuracy: {:.2%}, precision: {:.2%}, recall: {:.2%}".format(
    accuracy_score(Y_test, Y_pred),
    precision_score(Y_test, Y_pred, average='macro'),
    recall_score(Y_test, Y_pred, average='macro')))

#save_model(clf.fit(X, Y), 'products')
