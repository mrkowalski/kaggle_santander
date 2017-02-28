import sqlite3

import pandas as pd

asInt   = lambda x: None if x.endswith('NA') else int(x)

def asBool(b: str) -> bool:
    b = b.strip()
    if b == "1": return True
    elif b == "0": return False
    else: return None

conn = sqlite3.connect('santander_v2.db')
c = conn.cursor()
c.execute('''CREATE TABLE bank_data (
    fecha_dato              DATE,
    ncodpers                INTEGER,
    ind_empleado            CHARACTER(1),
    pais_residencia         VARCHAR(30),
    sexo                    CHARACTER(1),
    age                     SMALLINT,
    ind_nuevo               BOOLEAN,
    antiguedad              SMALLINT,
    indrel                  SMALLINT,
    indrel_1mes             SMALLINT,
    tiprel_1mes             CHARACTER(1),
    indresi                 BOOLEAN,
    indext                  BOOLEAN,
    canal_entrada           VARCHAR(30),
    indfall                 BOOLEAN,
    nomprov                 VARCHAR(100),
    ind_actividad_cliente   BOOLEAN,
    renta                   INTEGER,
    segmento                SMALLINT,
    ind_ahor_fin_ult1       BOOLEAN,
    ind_aval_fin_ult1       BOOLEAN,
    ind_cco_fin_ult1        BOOLEAN,
    ind_cder_fin_ult1       BOOLEAN,
    ind_cno_fin_ult1        BOOLEAN,
    ind_ctju_fin_ult1       BOOLEAN,
    ind_ctma_fin_ult1       BOOLEAN,
    ind_ctop_fin_ult1       BOOLEAN,
    ind_ctpp_fin_ult1       BOOLEAN,
    ind_deco_fin_ult1       BOOLEAN,
    ind_deme_fin_ult1       BOOLEAN,
    ind_dela_fin_ult1       BOOLEAN,
    ind_ecue_fin_ult1       BOOLEAN,
    ind_fond_fin_ult1       BOOLEAN,
    ind_hip_fin_ult1        BOOLEAN,
    ind_plan_fin_ult1       BOOLEAN,
    ind_pres_fin_ult1       BOOLEAN,
    ind_reca_fin_ult1       BOOLEAN,
    ind_tjcr_fin_ult1       BOOLEAN,
    ind_valo_fin_ult1       BOOLEAN,
    ind_viv_fin_ult1        BOOLEAN,
    ind_nomina_ult1         BOOLEAN,
    ind_nom_pens_ult1       BOOLEAN,
    ind_recibo_ult1         BOOLEAN)''')

#    fecha_alta              DATE,
#    ult_fec_cli_1t          DATE,
#    tipodom                 SMALLINT,
#    cod_prov                SMALLINT,
#    conyuemp                BOOLEAN,

c.execute('CREATE INDEX ix_ncodpers ON bank_data (ncodpers ASC)')

reader = pd.read_csv(
    '/home/ubuntu/santander/data/train_ver2.csv',
    sep=',',
    chunksize=1000000)

n = 0
for chunk in reader:
    chunk.drop(['tipodom', 'cod_prov', 'conyuemp', 'fecha_alta', 'ult_fec_cli_1t'], inplace=True, axis=1)
    c.executemany('INSERT INTO bank_data VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', chunk.as_matrix())
    n = n + 1
    print(n)

conn.commit()
conn.close()
