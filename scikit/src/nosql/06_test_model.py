import csv, sys, sqlite3, commons, io, csv

q = __import__('02_sql_script')

class Predictable:
    def __init__(self, ncodpers, csvdata, dbdata):
        self.ncodpers = ncodpers
        self.csvdata = csvdata
        self.dbdata = dbdata

    def isComplete(self): return self.dbdata is not None

    def asList(self): return [self.ncodpers] + self.dbdata + self.csvdata

def dbdata_to_map(dbdata): return {row[0]: list(row[1:]) for row in dbdata}

def zip_csv_with_db(csvdata, dbdata):
    dbdata = dbdata_to_map(dbdata)
    return [Predictable(d[0], d[1:], dbdata.get(d[0])) for d in csvdata]

def si_no(v):
    if v == 'N':
        return 0
    elif v == 'S':
        return 1
    return None

def segmento(v, flag):
    if v is not None:
        v = v.strip()
    if v == '01 - TOP':
        return int(flag == 1)
    elif v == '02 - PARTICULARES':
        return int(flag == 2)
    elif v == '03 - UNIVERSITARIO':
        return int(flag == 3)
    return None

def to_int(v):
    try:
        return int(float(v))
    except:
        if v == 'NA': return None
        return v

def read_test_csv():
    print("Loading test file... ", end='', flush=True)
    outputs = []
    with open('../input/test_ver2.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(reader)

        for row in reader:
            ncodpers = int(row[1].strip())
            o = [ncodpers]
            o.append(int(row[5]))  # age
            o.append(int(row[8]))  # antiguedad
            o.append(int(int(row[9]) == 99))  # indrel
            o.append(to_int(row[11]))  # indrel_1mes
            o.append(row[12])  # tiprel_1mes
            o.append(si_no(row[13]))  # indresi
            o.append(si_no(row[14]))  # indext
            o.append(row[17])  # indfall
            o.append(row[18])  # tipodom
            o.append(to_int(row[19]))  # cod_prov
            o.append(int(row[21]))  # ind_actividad_cliente
            o.append(segmento(row[23], 1))  # segmento
            o.append(segmento(row[23], 2))  # segmento
            o.append(segmento(row[23], 3))  # segmento
            outputs.append(o)
        print("Done.")
        return outputs

def make_dataframe(conn, chunk):
    cur = conn.cursor()
    print("\tChunk {}...Querying DB...".format(i))
    query = q.getquery_for_ncodpers('bd', 6, [c[0] for c in chunk])
    cur.execute(query)

    columns = [d[0] for d in cur.description]
    bd7_columns = ["bd7_" + c[4:] for c in columns if c.startswith("bd6_")]

    print("\tBuilding DF...".format(i))
    datarow = io.StringIO()
    csvwriter = csv.writer(datarow)
    predictions = zip_csv_with_db(chunk, cur.fetchall())
    # print("Missing data for ncodpers: {}".format(", ".join([str(p.ncodpers) for p in predictions if not p.isComplete()])))
    csvwriter.writerows([p.asList() for p in predictions if p.isComplete()])
    datarow.seek(0)
    return commons.read_csv(datarow, names=columns + bd7_columns)

def load_models(indicators):
    print("Loading models... ", end='', flush=True)
    models = {i: commons.read_model(i) for i in indicators}
    print("Done.")
    return models

def load_features(indicators):
    print("Loading features... ", end='', flush=True)
    features = {i: set(list(commons.read_dataframe_for_feature(i))) for i in indicators}
    print("Done.")
    return features

def customize_dataframe(df, features, indicator):
    df = df.drop("bd1_ncodpers", axis=1)
    df = df.drop(list(set(list(df)) - features), axis=1)
    df = df.drop("bd7_" + indicator, axis=1)
    return df

conn = sqlite3.connect('santander.db')
submission = {}
test_data = read_test_csv()
all_models = load_models(commons.indicators)
all_features = load_features(commons.indicators)

n = 10000
for i in range(0, len(test_data), n):

    chunk = test_data[i:i + n]

    for row in chunk:
        if row[0] not in submission:
            submission[row[0]] = []

    df = make_dataframe(conn, chunk)
    customers = df['bd1_ncodpers'].values.tolist()

    for indicator in commons.indicators:

        print("Processing: {}".format(indicator))

        features = all_features[indicator]
        model = all_models[indicator]
        customized_df = customize_dataframe(df, features, indicator)

        print("\tPredicting {}...".format(indicator))

        for decision in zip(customers, model.predict(customized_df)):
            if decision[1]:
                submission[decision[0]].append(indicator)

with open('submission.csv', 'w') as submission_file:
    print("ncodpers,added_products", file=submission_file)
    for k, v in submission.items():
        cur = cur = conn.cursor()
        qresult = cur.execute(q.getlatestproductsquery(k))
        columns = [d[0] for d in cur.description]
        present_indicators = set([t[1] for t in zip(qresult.fetchone(), columns) if t[0]])

        print("{},{}".format(k, " ".join(set(v) - present_indicators)), file=submission_file)

# conn.commit()
conn.close()
