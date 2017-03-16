import commons
import pandas as pd
import logging as log

def load_models(indicators):
    print("Loading models... ", end='', flush=True)
    models = {i: commons.read_model(i) for i in indicators}
    print("Done.")
    return models

chunks = 1
df = pd.DataFrame()
for n in range(1, chunks+1):
    log.info('Loading dataframe...#{}'.format(n))
    df = df.append(pd.read_hdf(commons.FILE_DF + "." + str(n), key='santander'))

df = df[df['is_test_data']]
df.drop(['is_test_data'], inplace=True, axis=1)
df.drop(commons.indicators_ignored, inplace=True, axis=1)
df.drop(commons.indicators ,inplace=True,axis=1)

all_models = load_models(commons.indicators)
submission = {}

log.info("{}".format(list(df)))
