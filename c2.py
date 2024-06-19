import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm

train_df = pd.read_csv('C:/Gdrive/data2/train.csv')
test_df = pd.read_csv('C:/Gdrive/data2/test.csv')
sample_submission = pd.read_csv('C:/Gdrive/data2/sample_submission.csv')
df_all = pd.concat([train_df, test_df], axis=0)

def one_hot_encoding(df):
    
    return_df = pd.get_dummies(df, drop_first=True)
    
    return return_df

def to_add_feature(df):
    
    df['EXT_123_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 3
    df['EXT_23_mean'] = (df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 2
    df['EXT_12_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2']) / 2
    df['EXT_13_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_3']) / 2
    df['EXT_23_sabun'] = abs(df['EXT_SOURCE_2'] - df['EXT_SOURCE_3'])
    df['EXT_12_sabun'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_2'])
    df['EXT_13_sabun'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_3'])
    
    df['CREDIT_ANNUITY'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['CREDIT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['INCOME_TOTAL_ANNUITY'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TOTAL_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    
    df['DAYS_BIRTH_365_OWN_CAR_AGE'] = (df['DAYS_BIRTH'] / 365) - df['OWN_CAR_AGE']

    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    return df

def to_drop(df):
    
    drop_list = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'REG_REGION_NOT_LIVE_REGION', 'LIVE_REGION_NOT_WORK_REGION']
    droped_df = df.drop(columns=drop_list)
    
    return droped_df

df_encoded = one_hot_encoding(df_all)
added_features_df = to_add_feature(df_encoded)
all_features_df = to_drop(added_features_df)

assert len(df_all) == len(df_encoded)
assert len(df_all) == len(added_features_df)
assert len(df_all) == len(all_features_df)

train = all_features_df[all_features_df.loc[:, 'SK_ID_CURR'] < 171202]
test = all_features_df[all_features_df.loc[:, 'SK_ID_CURR'] > 171201]

train_x = train.drop(columns=['TARGET', 'SK_ID_CURR'])
train_y = train['TARGET']
test_x = test.drop(columns=['TARGET', 'SK_ID_CURR'])

X = train_x.values
y = train_y.values

fold = StratifiedKFold(n_splits=8, shuffle=True, random_state=69)
cv = list(fold.split(X, y))

lgbm_best_param = {'reg_lambda': 1.1564659040946654, 'reg_alpha': 9.90877329623665, 'colsample_bytree': 0.5034991685866442, 'subsample': 0.6055998601661783, 'max_depth': 3, 'min_child_weight': 39.72586351155486, 'learning_rate': 0.08532489659779158}

def fit_lgbm(X, y, cv, params: dict=None, verbose=100):
    
    oof_preds = np.zeros(X.shape[0])

    if params is None:
        params = {}

    models = []

    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMClassifier(**params, random_state=71, n_estimators=10000)
        clf.fit(x_train, y_train, 
                eval_set=[(x_valid, y_valid)],  
                #early_stopping_rounds=100, 
                eval_metric='auc',
                #verbose=verbose
                )

        models.append(clf)
        oof_preds[idx_valid] = clf.predict_proba(x_valid, num_iteration=clf.best_iteration_)[:, 1]
        print('Fold %2d AUC : %.6f' % (i + 1, roc_auc_score(y_valid, oof_preds[idx_valid])))
    
    score = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % score) 
    return oof_preds, models

oof, models = fit_lgbm(X, y, cv=cv, params=lgbm_best_param)

pred = np.array([model.predict_proba(test_x.values)[:, 1] for model in models])
pred = np.mean(pred, axis=0)

submission = sample_submission.copy()
submission['TARGET'] = pred

submission.to_csv('C:/Gdrive/data2/3rd_place_solution.csv', index=False)