import logging, os
import pandas as pd
from sklearn.externals import joblib

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

FILE_TRAIN='/home/ubuntu/santander/data/train_ver2.csv'
FILE_TEST='/home/ubuntu/santander/data/test_ver2.csv'
FILE_DF='dataframe.hdf5'
FILE_DF_CLEAN='dataframe_clean.hdf5'

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

indicators = ['ind_cco_fin_ult1', 'ind_cno_fin_ult1',
              'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
              'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_plan_fin_ult1',
              'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
              'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

populations = {'ALAVA': 321417, 'ALBACETE': 400007, 'ALICANTE': 1945642, 'ALMERIA': 699329, 'ASTURIAS': 1068165, 'AVILA': 168825, 'BADAJOZ': 693729, 'BALEARS, ILLES': 1111674, 'BARCELONA': 5540925, 'BIZKAIA': 1156447, 'BURGOS': 371248, 'CACERES': 410275, 'CADIZ': 1238492, 'CANTABRIA': 591888, 'CASTELLON': 601699, 'CEUTA': 84180, 'CIUDAD REAL': 524962, 'CORDOBA': 802422, 'CORUÑA, A': 1138161, 'CUENCA': 211899, 'GIPUZKOA': 713818, 'GIRONA': 761632, 'GRANADA': 919319, 'GUADALAJARA': 257723, 'HUELVA': 520668, 'HUESCA': 226329, 'JAEN': 664916, 'LEON': 489752, 'LERIDA': 440915, 'LUGO': 346005, 'MADRID': 6495551, 'MALAGA': 1652999, 'MELILLA': 83679, 'MURCIA': 1472049, 'NAVARRA': 644447, 'OURENSE': 326724, 'PALENCIA': 168955, 'PALMAS, LAS': 1103850, 'PONTEVEDRA': 955050, 'RIOJA, LA': 322027, 'SALAMANCA': 345548, 'SANTA CRUZ DE TENERIFE': 1014829, 'SEGOVIA': 161702, 'SEVILLA': 1942115, 'SORIA': 93291, 'TARRAGONA': 810178, 'TERUEL': 142183, 'TOLEDO': 706407, 'VALENCIA': 2566474, 'VALLADOLID': 532284, 'ZAMORA': 955050, 'ZARAGOZA': 978638}

indicators_ignored = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_viv_fin_ult1', 'ind_hip_fin_ult1', 'ind_cder_fin_ult1']

relevant_dates = [pd.Timestamp('2015-06'), pd.Timestamp('2016-03'), pd.Timestamp('2016-04'), pd.Timestamp('2016-05'), pd.Timestamp('2016-06')]

def read_model(feature):
    if os.path.isfile("models/" + feature + ".pkl"):
        return joblib.load("models/" + feature + ".pkl")
    return None

#ind_hip_fin_ult1  - 80 activations
#ind_viv_fin_ult1  - 68 activations
#ind_cder_fin_ult1 - 24 activations
#ind_ahor_fin_ult1 - 4 activations
#ind_aval_fin_ult1 - 8 activations
