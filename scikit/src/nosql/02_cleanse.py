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

def add_history(df, column, num_months):
    log.info('Adding history for column {}...'.format(column))
    to_merge = df[['ncodpers', 'fecha_dato', column]]
    for m in range(1, num_months+1):
        log.info("Month {}...".format(m))
        log.info("All cases: {}".format(df.shape))
        merged = to_merge.copy()
        merged['fecha_dato'] = merged['fecha_dato'] + pd.DateOffset(months = m)
        merged.rename(columns={column: column + "_" + str(m)}, inplace=True)
        df = df.merge(merged.set_index(['ncodpers', 'fecha_dato']), how='left', left_index=True, right_index=True, copy=False)
    return df

def fix_age(df):
    log.info('Convert age to numeric...')
    df['age'] = pd.to_numeric(df['age'], downcast='integer', errors='coerce')
    log.info('Fixing age...')
    df['age'] = df['age'].where((df['age'] >= 20) & (df['age'] <= 99))
    df['age'].fillna(df['age'].mean(), inplace=True)

def fix_renta(df):
    log.info('Convert renta to numeric...')
    df['renta'] = pd.to_numeric(df['renta'], errors='coerce')

def fix_indicators(df):
    log.info('Convert indicators to 0/1')
    for ind in commons.indicators:
        df[ind] = pd.to_numeric(df[ind], errors='coerce', downcast='unsigned')
        df[ind].fillna(0, inplace=True)

def show_activation_stats(df):
    log.info("Activation stats:")
    df = df[~df['is_test_data']]
    log.info("All cases: {}".format(df.shape[0]))
    for ind in commons.indicators:
        log.info("{}: {}. Distinct={}".format(ind, sum(df["act_" + ind] == 1), df["act_" + ind].unique()))

def add_product_history(df, num_months):
    df_products = df[['ncodpers', 'fecha_dato'] + commons.indicators]
    for ind in commons.indicators:
        log.info("Adding product history for {}".format(ind))
        df_products = add_history(df_products, ind, num_months)
    df_products.drop(commons.indicators ,inplace=True,axis=1)
    return df.merge(df_products.set_index(['ncodpers', 'fecha_dato']), how='inner', left_index=True, right_index=True, copy=False)

log.info('Loading dataframe...')
df = pd.read_pickle(commons.FILE_DF)

log.info('Strip month...')
df['fecha_dato'] = df['fecha_dato'].str.slice(stop=7)

log.info('Convert fecha_dato to date...')
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format="%Y-%m", errors='raise')

log.info('Convert ncodpers to numeric...')
df['ncodpers'] = pd.to_numeric(df['ncodpers'], downcast='unsigned', errors='raise')

log.info('Indexing...')
df = df.set_index(['ncodpers', 'fecha_dato'], drop=False)

fix_age(df)
fix_renta(df)
fix_indicators(df)
#show_indicator_stats(df)

df = add_product_history(df, 1)
df = add_activations(df)
show_activation_stats(df)
df = add_history(df, 'renta', 2)

log.info('Storing cleaned-up dataframe...')
df.to_pickle(commons.FILE_DF_CLEAN)
