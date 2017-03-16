import commons
import pandas as pd
import logging as log

def load_models(indicators):
    print("Loading models... ", end='', flush=True)
    models = {i: commons.read_model(i) for i in indicators}
    print("Done.")
    return models

df = pd.read_hdf('dataframe.hdf5.test', key='santander')
df.drop(commons.indicators ,inplace=True,axis=1)
for i in range(1,11):
    df.drop([ind + "_" + str(i) for ind in commons.indicators], inplace=True, axis=1, errors='ignore')

log.info("Number of test cases: {}".format(df.shape[0]))

all_models = load_models(commons.indicators)
submission = {}

log.info("{}".format(list(df)))

customers = df['ncodpers'].values.tolist()
df.drop(['ncodpers'], inplace=True, axis=1)
for indicator in commons.indicators:
    log.info("Processing: {}".format(indicator))
    for decision in zip(customers, model.predict(df)):
        if decision[1]:
            submission[decision[0]].append(indicator)

with open('submission.csv', 'w') as submission_file:
    log.info("ncodpers,added_products", file=submission_file)
    for k, v in submission.items():
        cur = cur = conn.cursor()
        qresult = cur.execute(q.getlatestproductsquery(k))
        columns = [d[0] for d in cur.description]
        present_indicators = set([t[1] for t in zip(qresult.fetchone(), columns) if t[0]])

        print("{},{}".format(k, " ".join(set(v) - present_indicators)), file=submission_file)
