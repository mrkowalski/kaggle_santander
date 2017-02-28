import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from functools import partial
import re

num_months = 4
chunk_size = 1000000

indicators = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
              'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
              'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
              'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
              'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

column_prefix = 'bd'
main_dataframe = 'dataframe.pkl'
dataframe_dir = 'dataframes'
main_csv = 'stuff_6mo_all_indicators.csv'

def read_dataframe(): return pd.read_pickle(dataframe_dir + "/" + main_dataframe)

def read_dataframe_for_feature(feature): return pd.read_pickle(dataframe_dir + "/" + feature + ".pkl")

def save_dataframe_for_feature(df, feature): df.to_pickle(dataframe_dir + "/" + feature + ".pkl")

def save_dataframe(df): df.to_pickle(dataframe_dir + "/" + main_dataframe)

def predicted_column_name(name): return "{}{}_{}".format(column_prefix, num_months, name)

def save_model(clf, feature): joblib.dump(clf, "models/" + feature + ".pkl")

def read_model(feature): return joblib.load("models/" + feature + ".pkl")

def dtypes_range(name, upto, dtype): return {("bd{}_{}".format(i, name), dtype) for i in range(1, upto + 1)}

dtypes = {
    'bd1_renta': np.int32,
    'bd1_sexo': np.int8,
    'bd1_pais_residencia': np.int16,
    'bd1_canal_entrada': np.int16,
    'bd1_ind_nuevo': np.bool
}

dtypes.update(dtypes_range('segmento_individual', num_months, np.bool))
dtypes.update(dtypes_range('segmento_vip', num_months, np.bool))
dtypes.update(dtypes_range('segmento_graduate', num_months, np.bool))
for i in indicators: dtypes.update(dtypes_range(i, num_months, np.bool))
dtypes.update(dtypes_range('cod_prov', num_months, np.uint8))
dtypes.update(dtypes_range('ind_empleado', num_months, np.int8))
dtypes.update(dtypes_range('age', num_months, np.int))
dtypes.update(dtypes_range('indrel_99', num_months, np.bool))
dtypes.update(dtypes_range('ind_actividad_cliente', num_months, np.bool))
dtypes.update(dtypes_range('antiguedad', num_months, np.int))
dtypes.update(dtypes_range('tipodom', num_months, np.bool))
dtypes.update(dtypes_range('indfall', num_months, np.bool))
dtypes.update(dtypes_range('indext', num_months, np.bool))
dtypes.update(dtypes_range('indresi', num_months, np.bool))
dtypes.update(dtypes_range('indrel_1mes', num_months, np.int8))
dtypes.update(dtypes_range('tiprel_1mes', num_months, np.int8))

rx_prefix = re.compile('bd\\d_')
le = LabelEncoder()

field_values = {
    'sexo': LabelEncoder().fit(['V', 'H']),
    'pais_residencia': LabelEncoder().fit(
        ['ES', 'CA', 'CH', 'CL', 'IE', 'AT', 'NL', 'FR', 'GB', 'DE', 'DO', 'BE', 'AR', 'VE', 'US', 'MX',
         'BR', 'IT', 'EC', 'PE', 'CO', 'HN', 'FI', 'SE', 'AL', 'PT', 'MZ', 'CN', 'TW', 'PL', 'IN', 'CR',
         'NI', 'HK', 'AD', 'CZ', 'AE', 'MA', 'GR', 'PR', 'RO', 'IL', 'RU', 'GT', 'GA', 'NO', 'SN',
         'MR', 'UA', 'BG', 'PY', 'EE', 'SV', 'ET', 'CM', 'SA', 'CI', 'QA', 'LU', 'PA', 'BA', 'BO', 'AU',
         'BY', 'KE', 'SG', 'HR', 'MD', 'SK', 'TR', 'AO', 'CU', 'GQ', 'EG', 'ZA', 'DK', 'UY', 'GE',
         'TH', 'DZ', 'LB', 'JP', 'NG', 'PK', 'TN', 'TG', 'KR', 'GH', 'RS', 'VN', 'PH', 'KW', 'NZ',
         'MM', 'KH', 'GI', 'SL', 'GN', 'GW', 'OM', 'CG', 'LV', 'LT', 'ML', 'MK', 'HU', 'IS', 'LY', 'CF',
         'GM', 'KZ', 'CD', 'BZ', 'ZW', 'DJ', 'JM', 'BM', 'MT'
         ]),
    'ind_empleado': LabelEncoder().fit(['N', 'A', 'B', 'F', 'S']),
    'canal_entrada': LabelEncoder().fit(
        ['KHL', 'KHE', 'KHD', 'KFA', 'KFC', 'KAT', 'KAZ', 'RED', 'KHC', 'KHK', 'KGN', 'KHM', 'KHO', 'KDH',
         'KEH', 'KAD', 'KBG', 'KGC', 'KHF', 'KFK', 'KHN', 'KHA', 'KAF', 'KGX', 'KFD', 'KAG', 'KFG', 'KAB',
         'KCC', 'KAE', 'KAH', 'KAR', 'KFJ', 'KFL', 'KAI', 'KFU', 'KAQ', 'KFS', 'KAA', 'KFP', 'KAJ', 'KFN',
         'KGV', 'KGY', 'KFF', 'KAP', 'KDE', 'KFV', '013', 'K00', 'KAK', 'KCK', 'KCL', 'KAY', 'KBU', 'KDR',
         'KAC', 'KDT', 'KCG', 'KDO', 'KDY', 'KBQ', 'KDA', 'KBO', 'KCI', 'KEC', 'KBZ', 'KES', 'KDX', 'KAS',
         '007', 'KEU', 'KCA', 'KAL', 'KDC', 'KAW', 'KCS', 'KCB', 'KDU', 'KDQ', 'KCN', 'KCM', '004', 'KCH',
         'KCD', 'KCE', 'KEV', 'KBL', 'KEA', 'KBH', 'KDV', 'KFT', 'KEY', 'KAO', 'KEJ', 'KEO', 'KEI', 'KEW',
         'KDZ', 'KBV', 'KBR', 'KBF', 'KDP', 'KCO', 'KCF', 'KCV', 'KAM', 'KEZ', 'KBD', 'KAN', 'KBY', 'KCT',
         'KDD', 'KBW', 'KCU', 'KBX', 'KDB', 'KBS', 'KBE', 'KCX', 'KBP', 'KBN', 'KEB', 'KDS', 'KEL', 'KDG',
         'KDF', 'KEF', 'KCP', 'KDM', 'KBB', 'KDW', 'KBJ', 'KFI', 'KBM', 'KEG', 'KEN', 'KEQ', 'KAV', 'KFH',
         'KFM', 'KAU', 'KED', 'KFR', 'KEK', 'KFB', 'KGW', 'KFE', 'KGU', 'KDI', 'KDN', 'KEE', 'KCR', 'KCQ',
         'KEM', 'KCJ', 'KHQ', 'KDL', '025', 'KHP', 'KHR', 'KHS']),
    'indrel_1mes': LabelEncoder().fit(['1', '2', '3', '4', 'P']),
    'tiprel_1mes': LabelEncoder().fit(['A', 'I', 'P', 'R', 'N']),
    'indfall': LabelEncoder().fit(['N', 'S'])
}

def col_to_int(range, v):
    if v is None:
        return -1
    else:
        v = v.strip()
        if len(v) == 0:
            return -1
        else:
            return field_values[range].transform([v])[0]

def col_to_intvalue(default, v):
    if v is None:
        return default
    else:
        try:
            return int(v)
        except:
            return default

def feature_range_list(name, upto):  return ["bd{}_{}".format(i, name) for i in range(1, upto + 1)]

cols_as_integers = set(['segmento_', 'cod_prov', 'ind_', 'ind_empleado', 'age'])
cols_to_convert = ['bd1_sexo', 'bd1_pais_residencia', 'bd1_ind_empleado', 'bd1_canal_entrada'] + feature_range_list(
    'indrel_1mes', num_months) + feature_range_list('indfall', num_months) + feature_range_list(
    'segmento_individual', num_months) + feature_range_list('tiprel_1mes', num_months) + feature_range_list(
    'segmento_graduate', num_months) + feature_range_list(
    'segmento_vip', num_months) + feature_range_list('cod_prov', num_months) + feature_range_list('age', num_months)
for i in indicators: cols_to_convert = cols_to_convert + feature_range_list(i, num_months)

def make_converters():
    converters = {}
    for c in cols_to_convert:
        if rx_prefix.match(c[:4]):
            if any([c[4:].startswith(token) for token in cols_as_integers]):
                converters[c] = partial(col_to_intvalue, 0)
            else:
                converters[c] = partial(col_to_int, c[4:])
        else:
            converters[c] = partial(col_to_int, c)
    return converters

def fillna_range(data, name, upto, v):
    for i in range(1, upto + 1):
        data["bd{}_{}".format(i, name)].fillna(v, inplace=True)

def read_csv(source, names=None): return pd.read_csv(source, sep=',', nrows=chunk_size, converters=make_converters(),
                                                     dtype=dtypes, names=names)
