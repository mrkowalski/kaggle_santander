import commons, sys
import logging as log
import pandas as pd
import numpy as np

def add_history(df, column, num_months):
    log.info('Adding history for column {}...'.format(column))
    to_merge = df[['ncodpers', 'fecha_dato', column]].copy()
    for m in range(1, num_months+1):
        merged = to_merge.copy()
        merged['fecha_dato'] = merged['fecha_dato'] + pd.DateOffset(months = m)
        merged.rename(index=str, columns={column: column + "_" + str(m)}, inplace=True)
        df = df.merge(merged, on=['ncodpers', 'fecha_dato'], how='left')
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

log.info('Loading dataframe...')
df = pd.read_pickle(commons.FILE_DF)

log.info('Strip month...')
df['fecha_dato'] = df['fecha_dato'].str.slice(stop=7)

log.info('Convert fecha_dato to date...')
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format="%Y-%m", errors='raise')

log.info('Convert ncodpers to numeric...')
df['ncodpers'] = pd.to_numeric(df['ncodpers'], downcast='unsigned', errors='raise')

fix_age(df)
fix_renta(df)

df = add_history(df, 'renta', 2)

log.info('Storing cleaned-up dataframe...')
df.to_pickle(commons.FILE_DF_CLEAN)
