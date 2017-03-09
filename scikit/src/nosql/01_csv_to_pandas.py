import commons, logging as log
import pandas as pd

def load(f, is_test_data):
    log.info('Loading {}...'.format(f))
    df = pd.read_csv(f, sep=',')
    df.drop(['tipodom', 'cod_prov', 'conyuemp', 'fecha_alta', 'ult_fec_cli_1t'], inplace=True, axis=1)
    df.insert(0, 'is_test_data', is_test_data)
    return df

df = pd.concat([load(commons.FILE_TRAIN, False), load(commons.FILE_TEST, True)])

log.info("Pickling...")
df.to_pickle(commons.FILE_DF)
