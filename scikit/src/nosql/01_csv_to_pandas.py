import sys, commons, logging as log
import pandas as pd

def load(f, is_test_data):
    log.info('Loading {}...'.format(f))
    df = pd.read_csv(f, sep=',')
    df.drop(['tipodom', 'cod_prov', 'conyuemp', 'fecha_alta', 'ult_fec_cli_1t'], inplace=True, axis=1)
    df.insert(0, 'is_test_data', is_test_data)
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

df = pd.concat([load(commons.FILE_TRAIN, False), load(commons.FILE_TEST, True)])

log.info('Strip month...')
df['fecha_dato'] = df['fecha_dato'].str.slice(stop=7)

log.info('Convert fecha_dato to date...')
df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format="%Y-%m", errors='raise')

log.info('Convert ncodpers to numeric...')
df['ncodpers'] = pd.to_numeric(df['ncodpers'], downcast='unsigned', errors='raise')

#log.info('Indexing...')
#df = df.set_index(['ncodpers', 'fecha_dato'], drop=False)

#log.info("Non-unique: {}".format(df[df.duplicated(['ncodpers', 'fecha_dato'])], keep=False))

log.info("Indices: {}".format(df.index.names))
log.info('Convert antiguedad to numeric...')
df['antiguedad'] = pd.to_numeric(df['antiguedad'], downcast='unsigned', errors='coerce')

fix_age(df)
fix_renta(df)
fix_indicators(df)
fix_indrel_1mes(df)

badtypes=['antiguedad', 'canal_entrada', 'nomprov', 'pais_residencia', 'tiprel_1mes']
#[('canal_entrada', dtype('O')), ('fecha_dato', dtype('<M8[ns]')), ('nomprov', dtype('O')), ('pais_residencia', dtype('O'))]

as_cat(df, 'sexo')
as_cat(df, 'segmento')
as_cat(df, 'indext')
as_cat(df, 'indfall')
as_cat(df, 'ind_empleado')
as_cat(df, 'indresi')
as_cat(df, 'indrel_1mes')
as_cat(df, 'tiprel_1mes')

#Some less suitable ones...
as_cat(df, 'nomprov')
as_cat(df, 'pais_residencia')
as_cat(df, 'canal_entrada')

#log.info("antiguedad: {}".format(df['antiguedad'].value_counts(dropna=False)))

chunks = 5
for n in range(1, chunks+1):
    df_n = df[(df['ncodpers'] % chunks == 0)]
    df_n = add_product_history(df_n, 5)
    log.info("Pickling...#{}".format(n))
    df_n.to_hdf(commons.FILE_DF + "." + str(n), key='santander', mode='w', format='fixed')

#df02 = df[(df['ncodpers'] % 2 == 1)]
#log.info("{}".format(df01.shape))
#log.info("{}".format(df02.shape))
#sys.exit()

