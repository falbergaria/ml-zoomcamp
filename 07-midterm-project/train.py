import pandas as pd

from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

import pickle


output_file = 'model.bin'

df = pd.read_csv('train.csv')


df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

del df['id']


df_full_train = df.reset_index(drop=True)

y_full_train = df['pitnextlap'].astype('int').values

del df_full_train['pitnextlap']


dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)


features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)


xgb_params = {
        'eta': 0.3, 
        'max_depth': 8,
        'min_child_weight': 20,
        
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

model = xgb.train(xgb_params, dtrain, num_boost_round=200)


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
