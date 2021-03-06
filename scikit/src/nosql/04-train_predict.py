import commons, sys
import pandas as pd
import logging as log

def load_models(indicators):
    print("Loading models... ", end='', flush=True)
    models = {i: commons.read_model(i) for i in indicators}
    print("Done.")
    return models

df = pd.read_hdf('dataframe.hdf5.test', key='santander')
df.drop(commons.indicators, inplace=True, axis=1)

log.info("Number of test cases: {}".format(df.shape[0]))

all_models = load_models(commons.indicators)
submission = {}

log.info("{}".format(list(df)))

customers = df['ncodpers'].values.tolist()
df.drop(['ncodpers'], inplace=True, axis=1)
for indicator in commons.indicators:
    log.info("Processing: {}".format(indicator))
    model = all_models[indicator]
    for decision in zip(customers, model.predict_proba(df)):
        if decision[0]  not in submission: submission[decision[0]] = list()
        if decision[1][1] >= decision[1][0]: submission[decision[0]].append((indicator, decision[1][1]))

with open('submission.csv', 'w') as submission_file:
    print("ncodpers,added_products", file=submission_file)
    for k, v in submission.items():
        print("{},{}".format(k, " ".join([i[0] for i in sorted(v, reverse=True, key=lambda e: e[1])])), file=submission_file)
