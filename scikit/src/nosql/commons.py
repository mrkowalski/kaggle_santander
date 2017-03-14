import logging
import pandas as pd

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

FILE_TRAIN='/home/ubuntu/santander/data/train_ver2.csv'
FILE_TEST='/home/ubuntu/santander/data/test_ver2.csv'
FILE_DF='dataframe.hdf5'
FILE_DF_CLEAN='dataframe_clean.hdf5'

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

indicators = ['ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
              'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
              'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_plan_fin_ult1',
              'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
              'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

indicators_ignored = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_viv_fin_ult1', 'ind_hip_fin_ult1']

#ind_hip_fin_ult1  - 80 activations
#ind_viv_fin_ult1  - 68 activations
#ind_ahor_fin_ult1 - 4 activations
#ind_aval_fin_ult1 - 8 activations
