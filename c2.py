import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm

from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

import datetime
import time
last_time = None  # 前回の時間を保持するグローバル変数

def current_timestamp():
    # 現在のタイムスタンプを返す関数
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def print_with_timestamp(message):
    # メッセージの前にタイムスタンプを付けて出力する関数
    global last_time
    current_time = time.time()
    if last_time is not None:
        elapsed_time = current_time - last_time
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}, 処理時間: {elapsed_time:.2f}秒")
    else:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    last_time = current_time

import sys
import warnings
warnings.filterwarnings('ignore')
# ログファイルを開く
log_file = open("C:/Gdrive/data2/results.log", "w")
# 標準出力をログファイルにリダイレクトする
sys.stdout = log_file

train_df = pd.read_csv('C:/Gdrive/data2/train.csv')
test_df = pd.read_csv('C:/Gdrive/data2/test.csv')
sample_submission = pd.read_csv('C:/Gdrive/data2/sample_submission.csv')
df_all = pd.concat([train_df, test_df], axis=0)

def one_hot_encoding(df):
    
    return_df = pd.get_dummies(df, drop_first=True)
    
    return return_df

def to_add_feature(df):

    # 異常値を欠損値に（365243 -> np.nan）
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
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

    # Kaggle 1st solution を参考に追加

    # credit_annuity_ratio: AMT_CREDIT / AMT_ANNUITY の比率を計算
    df['credit_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    
    # credit_goods_price_ratio: AMT_CREDIT / AMT_GOODS_PRICE の比率を計算
    df['credit_goods_price_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    # credit_downpayment: AMT_GOODS_PRICE - AMT_CREDIT を計算
    df['credit_downpayment'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    
    # AGE_INT: DAYS_BIRTH を年単位の整数値に変換
    df['AGE_INT'] = (df['DAYS_BIRTH'] / -365).astype(int)
    
    # region_id: REGION_RATING_CLIENT をカテゴリカル変数として扱うため、ラベルエンコーディング
    df['region_id'] = pd.factorize(df['REGION_RATING_CLIENT'])[0]

    return df


def add_features_parallel(df, n_neighbors=500):
    # neighbors_target_mean_500: 各行について、EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, 
    # credit_annuity_ratio の4つの特徴量を使って最近傍の500件を見つけ、それらのTARGETの平均値を計算

    features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'credit_annuity_ratio']
    df_neighbors = df[features].fillna(0)
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(df_neighbors)
    distances, indices = nbrs.kneighbors(df_neighbors)
    
    def get_neighbor_mean(i):
        return df.loc[df.index[indices[i]]]['TARGET'].mean()
    
    neighbor_means = Parallel(n_jobs=-1)(delayed(get_neighbor_mean)(i) for i in range(len(indices)))
    df['neighbors_target_mean_500'] = neighbor_means

    # SK_ID_CURRとneighbors_target_mean_500の結果をCSVファイルに保存
    result_df = pd.DataFrame({'SK_ID_CURR': df['SK_ID_CURR'], 'neighbors_target_mean_500': df['neighbors_target_mean_500']})
    result_df.to_csv('C:/Gdrive/data2/neighbors_target_mean_500.csv', index=False)
    
    return df

def add_neighbors_target_mean_500(df):
    # 保存したCSVファイルを読み込む
    neighbors_target_mean_500_df = pd.read_csv('C:/Gdrive/data2/neighbors_target_mean_500.csv')
    
    # データフレームにneighbors_target_mean_500を追加
    df = pd.merge(df, neighbors_target_mean_500_df, on='SK_ID_CURR', how='left')
    
    return df

def to_drop(df):
    
    drop_list = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'REG_REGION_NOT_LIVE_REGION', 'LIVE_REGION_NOT_WORK_REGION']
    droped_df = df.drop(columns=drop_list)
    
    return droped_df

print_with_timestamp('dfをone-hot encodingします...')
df_encoded = one_hot_encoding(df_all)
print_with_timestamp('one-hot encoding完了')

print_with_timestamp('主要な特徴量を追加します...')
added_features_df = to_add_feature(df_encoded)
print_with_timestamp('主要な特徴量追加完了')

#通常時、ここの3行はコメントアウト可能（計算が早くなる）
print_with_timestamp('neighbors_target_mean_500を計算して追加します...') 
added_features_df = add_features_parallel(added_features_df) 
print_with_timestamp('neighbors_target_mean_500計算追加完了') 

print_with_timestamp('neighbors_target_mean_500を外部ファイルから追加します...')
added_features_df = add_neighbors_target_mean_500(added_features_df)
print_with_timestamp('neighbors_target_mean_500追加完了')

print_with_timestamp('不要な特徴量を削除します...')
all_features_df = to_drop(added_features_df)
print_with_timestamp('特徴量削除完了')

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

print_with_timestamp('LightGBMで学習を行います...')
oof, models = fit_lgbm(X, y, cv=cv, params=lgbm_best_param)
print_with_timestamp('学習完了')

pred = np.array([model.predict_proba(test_x.values)[:, 1] for model in models])
pred = np.mean(pred, axis=0)

submission = sample_submission.copy()
submission['TARGET'] = pred

print_with_timestamp('予測結果を出力します...')
submission.to_csv('C:/Gdrive/data2/3rd_place_solution.csv', index=False)
print_with_timestamp('出力完了')

# ログファイルを閉じる
log_file.close()