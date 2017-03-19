import sys, commons, logging as log, math
import pandas as pd
import numpy as np

def load(f, is_test_data):
    log.info('Loading {}...'.format(f))
    df = pd.read_csv(f, sep=',', nrows=50000)
    df.drop(['tipodom', 'cod_prov', 'conyuemp', 'fecha_alta', 'ult_fec_cli_1t'], inplace=True, axis=1)
    df.insert(0, 'is_test_data', is_test_data)
    return df

def fix_nulls(df, feature):
    log.info("Fixing {} with same-client-non-nulls".format(feature))
    df[['ncodpers', feature]].copy().set_index('ncodpers').to_dict()
    all_vals = {}
    for d in pd.date_range('2015-01', '2016-07', freq='MS'):
        vals = df[df['fecha_dato'] == d][['ncodpers', feature]].copy().set_index('ncodpers').to_dict()[feature]
        vals = {e[0]:e[1] for e in vals.items() if pd.notnull(e[1]) }
        all_vals = {**all_vals, **vals}
    nulls_before=df[df[feature].isnull()].shape[0]
    df[feature] = df['ncodpers'].map(all_vals)
    log.info("Nulls for {}: before: {}, after: {}".format(feature, nulls_before, df[df[feature].isnull()].shape[0]))

def fix_age(df):
    log.info('Convert age to numeric...')
    df['age'] = pd.to_numeric(df['age'], downcast='integer', errors='coerce')
    fix_nulls(df, 'age')
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

def add_product_history(df, num_months):
    df_products = df[['ncodpers', 'fecha_dato'] + commons.indicators].copy()
    for ind in commons.indicators:
        log.info("Adding product history for {}".format(ind))
        df_products = add_history(df_products, ind, num_months)
    df_products.drop(commons.indicators ,inplace=True,axis=1)
    return df.merge(df_products, how='inner', copy=False, on=['ncodpers', 'fecha_dato'])

def add_history(df, column, num_months):
    log.info('Adding history for column {}...'.format(column))
    to_merge = df[['ncodpers', 'fecha_dato', column]]
    for m in range(1, num_months+1):
        log.info("Month {}...".format(m))
#        log.info("All cases: {}".format(df.shape))
        merged = to_merge.copy()
        merged['fecha_dato'] = merged['fecha_dato'] + pd.DateOffset(months = m)
        merged.rename(columns={column: column + "_" + str(m)}, inplace=True)
        #df = df.merge(merged.set_index(['ncodpers', 'fecha_dato']), how='left', left_index=True, right_index=True, copy=False)
        df = df.merge(merged, how='left', copy=False, on=['ncodpers', 'fecha_dato'])
    return df

def as_cat(df, col):
    c = df[col].astype('category')
    log.info("{} categories: {}".format(col, c.cat.categories))
    df[col] = c.cat.codes

def fix_indrel_1mes0(v):
    if pd.isnull(v): return v
    v = str(v).strip()
    if v == '1.0': v = '1'
    elif v == '2.0': v = '2'
    elif v == '3.0': v = '3'
    elif v == '4.0': v = '4'
    return v

def fix_indrel_1mes(df):
    log.info("Fixing indrel_1mes...")
    df['indrel_1mes'] = df['indrel_1mes'].map(fix_indrel_1mes0)

def product_history_in_chunks0(df, start, end, months):
    log.info("Processing history chunk, start={}, end={}".format(start, end))
    return add_product_history(df[(df['ncodpers'] >= start) & (df['ncodpers'] < end)], months)

def product_history_in_chunks(df, chunks, months):
    qs = [df['ncodpers'].quantile(x / chunks) for x in range(1, chunks)]
    in_chunks = [product_history_in_chunks0(df, q[0], q[1], months) for q in zip([0] + qs, qs + [math.inf])]

    log.info("History chunk sizes: {}".format([f.shape[0] for f in in_chunks]))

    df = in_chunks[0]
    del in_chunks[0]

    while len(in_chunks) > 0:
        df = df.append(in_chunks[0])
        del in_chunks[0]
    return df

def fix_nomprov(df):
    log.info('Transofrming nomprov...')
    fix_nulls(df, 'nomprov')
    as_cat(df, 'nomprov')
    df['is_big_city'] = df['nomprov'].isin(['MADRID', 'BARCELONA'])
    df['is_africa'] = df['nomprov'].isin(['CEUTA', 'MELILLA'])
    df['nomprov'] = df['nomprov'].map(commons.populations)

def fix_pais_residencia(df):
    log.info('Fixing pais_residencia...')
    fix_nulls(df, 'pais_residencia')
    df['pais_residencia'].fillna('ES')
    as_cat(df, 'pais_residencia')

def as_dummy(df, feature):
    df = df.join(pd.get_dummies(df[feature], prefix=feature))
    df.drop(feature, axis=1, inplace=True)
    return df

df = pd.concat([load(commons.FILE_TRAIN, False), load(commons.FILE_TEST, True)])
#df = load(commons.FILE_TEST, True)
for ind in commons.indicators:
    if ind not in df.columns:
        df[ind] = None

df.drop(commons.indicators_ignored, inplace=True, axis=1, errors='ignore')

log.info('Strip month...')
df['fecha_month'] = pd.to_numeric(df['fecha_dato'].str.slice(start=5, stop=7))
df['fecha_dato'] = df['fecha_dato'].str.slice(stop=7)

log.info('Convert fecha_dato to date...')
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format="%Y-%m", errors='raise')

log.info('Convert ncodpers to numeric...')
df['ncodpers'] = pd.to_numeric(df['ncodpers'], downcast='unsigned', errors='raise')

#log.info('Indexing...')
#df = df.set_index(['ncodpers', 'fecha_dato'], drop=False)

#log.info("Non-unique: {}".format(df[df.duplicated(['ncodpers', 'fecha_dato'])], keep=False))

log.info('Convert antiguedad to numeric...')
df['antiguedad'] = pd.to_numeric(df['antiguedad'], downcast='unsigned', errors='coerce')
fix_nulls(df, 'antiguedad')

fix_age(df)
fix_renta(df)
fix_indicators(df)
fix_indrel_1mes(df)
fix_pais_residencia(df)
fix_nomprov(df)

as_cat(df, 'sexo')
as_cat(df, 'segmento')
as_cat(df, 'indext')
as_cat(df, 'indfall')
as_cat(df, 'ind_empleado')
as_cat(df, 'indresi')
as_cat(df, 'indrel_1mes')
as_cat(df, 'tiprel_1mes')

#Some less suitable ones...
as_cat(df, 'canal_entrada')

df = product_history_in_chunks(df, 10, 5)

log.info("Keeping only relevant dates...")
df = df[df['fecha_dato'].isin([pd.Timestamp('2015-06'), pd.Timestamp('2016-03'), pd.Timestamp('2016-04'), pd.Timestamp('2016-05'), pd.Timestamp('2016-06')])]

df = as_dummy(df, 'ind_empleado')
df = as_dummy(df, 'segmento')
df = as_dummy(df, 'indrel_1mes')
df = as_dummy(df, 'tiprel_1mes')

print(str(list(df)))

df['fecha_dato'] = df['fecha_month'].copy()
df.drop(['fecha_month'], inplace=True, axis=1)

chunks = 5
df_train_data = df[~df['is_test_data']].copy()
df_train_data.drop(['is_test_data'], inplace=True, axis=1)
if df_train_data.shape[0] > 0:
    for n, df_n in df_train_data.groupby(np.arange(len(df_train_data)) // (df_train_data.shape[0] // chunks)):
        n = n + 1
        log.info("Pickling...#{}".format(n))
        df_n.to_hdf(commons.FILE_DF + "." + str(n), key='santander', mode='w', format='fixed')

df_test_data = df[df['is_test_data']].copy()
df_test_data.drop(['is_test_data'], inplace=True, axis=1)
if df_test_data.shape[0] > 0:
    df_test_data.to_hdf(commons.FILE_DF + ".test", key='santander', mode='w', format='fixed')
