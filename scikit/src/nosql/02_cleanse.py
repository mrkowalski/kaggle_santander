import commons, sys
import logging as log
import pandas as pd
import numpy as np

def add_activations(df):
    for ind in commons.indicators:
        log.info("Adding activations for {}".format(ind))
        ind_prev = ind + "_1"
        res = df[ind].sub(df[ind_prev])
        res[res < 0] = 0
        df = df + res.fillna(0).rename("act_" + ind)
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

log.info("{}".format(list(df)))
df = add_activations(df)
#show_activation_stats(df)

#log.info('Storing cleaned-up dataframe...')
#df.to_pickle(commons.FILE_DF_CLEAN)
