#BUG!!!! WHAT ABOUT FIRST MONTH? IT SHOULD MOST LIKELY CONTAIN NO ACTIVATION FLAGS.
import commons, sys, os
import logging as log
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.externals import joblib

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

chunks = 4
df = pd.DataFrame()
for n in range(1, chunks+1):
    log.info('Loading dataframe...#{}'.format(n))
    df = df.append(pd.read_hdf(commons.FILE_DF + "." + str(n), key='santander'))

df.drop(commons.indicators_ignored, inplace=True, axis=1)
df = add_activations(df)
df.drop(commons.indicators ,inplace=True,axis=1)
df.drop([i + "_1" for i in commons.indicators] ,inplace=True,axis=1)
df.drop(['ncodpers', 'fecha_dato'], inplace=True, axis=1)

log.info("Creating activation column...")
activation_columns=["act_" + i for i in commons.indicators]
df = df[df[activation_columns].sum(axis=1) != 0]
#show_activation_stats(df)
df['activation'] = df.apply(lambda r: "".join([str(int(e)) for e in r[activation_columns]]), axis=1)
df['activation'] = df['activation'].astype('category')
log.info("{}".format(df['activation'].cat.categories))
sys.exit()

#Build activations model
log.info("Building activation model...")
df_any_activation = df.drop(activation_columns, axis=1)
X = df_any_activation.drop('activation', axis=1)
Y = df['activation']
clf = xgb.XGBClassifier()
log.info("Training activation classifier...")
save_model(clf.fit(X, Y), 'activation')

#Keep only records with any activations
df = df[df['activation']]
df = df.drop('activation')
log.info("{}".format(df.shape))
X = df.drop(["act_" + i for i in commons.indicators], axis=1)
for ind in commons.indicators:
    log.info("Building model for: {}".format(ind))
    feature = "act_" + ind
    Y = df[feature]
    clf = xgb.XGBClassifier()
    save_model(clf.fit(X, Y), feature)

#log.info('Storing cleaned-up dataframe...')
#df.to_pickle(commons.FILE_DF_CLEAN)
