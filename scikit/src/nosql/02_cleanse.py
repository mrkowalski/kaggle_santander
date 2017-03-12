#BUG!!!! WHAT ABOUT FIRST MONTH? IT SHOULD MOST LIKELY CONTAIN NO ACTIVATION FLAGS.
import commons, sys, os
import logging as log
import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib

def save_model(clf, feature):
    log.info("Saving model: {}".format(feature))
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

df = add_activations(df)
df.drop(commons.indicators ,inplace=True,axis=1)
df.drop([i + "_1" for i in commons.indicators] ,inplace=True,axis=1)
#show_activation_stats(df)

log.info("{}".format([i for i in list(zip(df, df.dtypes)) if i[1] not in ('float64', 'int8', 'uint32', 'bool')]))

for ind in commons.indicators:
    feature = "act_" + ind
    X = df.drop(["act_" + i for i in commons.indicators] + ['ncodpers', 'fecha_dato'], axis=1)
    Y = df[feature]
    clf = xgb.XGBClassifier()
    save_model(clf.fit(X, Y), feature)

#log.info('Storing cleaned-up dataframe...')
#df.to_pickle(commons.FILE_DF_CLEAN)
