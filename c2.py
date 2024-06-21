import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm
from lightgbm import early_stopping
from lightgbm import log_evaluation

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
class DualLogger:
    def __init__(self, filepath, terminal):
        self.terminal = terminal
        self.log = open(filepath, "a")  # 追記モードでファイルを開く

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # このメソッドは、ファイルやターミナルがフラッシュを必要とする場合に呼ばれます
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # リソースを適切にクリーンアップ
        self.log.close()

def one_hot_encoding(df):
    
    return_df = pd.get_dummies(df, drop_first=True)
    
    return return_df

def to_add_feature(df):

    # 異常値を欠損値に（365243 -> np.nan）
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # df['EXT_123_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 3
    df['EXT_23_mean'] = (df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 2
    df['EXT_12_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2']) / 2
    df['EXT_13_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_3']) / 2
    df['EXT_23_sabun'] = abs(df['EXT_SOURCE_2'] - df['EXT_SOURCE_3'])
    df['EXT_12_sabun'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_2'])
    df['EXT_13_sabun'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_3'])
    
    # df['CREDIT_ANNUITY'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # df['CREDIT_GOODS_PRICE'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # df['INCOME_TOTAL_ANNUITY'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # df['INCOME_TOTAL_CREDIT'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    
    df['DAYS_BIRTH_365_OWN_CAR_AGE'] = (df['DAYS_BIRTH'] / 365) - df['OWN_CAR_AGE']

    # Kaggle 1st solution を参考に追加
    # (Ryan)
    # EXT_SOURCE_3による除算
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        df[f'{col}_DIV_EXT_SOURCE_3'] = df[col] / df['EXT_SOURCE_3']
    # 日付関連の特徴量
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # 年齢層
    df['AGE_RANGE'] = pd.cut(df['DAYS_BIRTH'] / -365, bins=[0, 20, 30, 40, 50, 60, np.inf], labels=[1, 2, 3, 4, 5, 6])
    # 雇用期間の比率
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # クレジットに対する収入の比率
    df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    # 家族構成に関する特徴量
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']

    # (Phil)
    # credit_annuity_ratio: AMT_CREDIT / AMT_ANNUITY の比率を計算
    df['credit_annuity_ratio'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    # credit_goods_price_ratio: AMT_CREDIT / AMT_GOODS_PRICE の比率を計算
    df['credit_goods_price_ratio'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # AGE_INT: DAYS_BIRTH を年単位の整数値に変換
    df['AGE_INT'] = (df['DAYS_BIRTH'] / -365).astype(int)
    # region_id: REGION_RATING_CLIENT をカテゴリカル変数として扱うため、ラベルエンコーディング
    df['region_id'] = pd.factorize(df['REGION_RATING_CLIENT'])[0]

    # (Yang)
    # 収入、支払い、時間に関連する KPI
    df['INCOME_ANNUITY_RATIO'] = df['AMT_INCOME_TOTAL'] / df['AMT_ANNUITY']
    # 時間加重平均（DAYSを重みとして使用）
    df['WEIGHTED_ANNUITY'] = df['AMT_ANNUITY'] * (1 / (1 + np.abs(df['DAYS_EMPLOYED'])))
    df['WEIGHTED_CREDIT'] = df['AMT_CREDIT'] * (1 / (1 + np.abs(df['DAYS_EMPLOYED'])))
    # 収入と時間の関係
    df['INCOME_PER_EMPLOYED_DAY'] = df['AMT_INCOME_TOTAL'] / (1 + np.abs(df['DAYS_EMPLOYED']))
    df['CREDIT_PER_EMPLOYED_DAY'] = df['AMT_CREDIT'] / (1 + np.abs(df['DAYS_EMPLOYED']))
    # 信用履歴の長さに関する特徴量
    df['CREDIT_HISTORY_LENGTH'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
    df['CREDIT_HISTORY_LENGTH_RATIO'] = df['CREDIT_HISTORY_LENGTH'] / df['DAYS_BIRTH']
    # 家族構成と収入の関係
    df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['CREDIT_PER_FAMILY_MEMBER'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    # EXT_SOURCE の組み合わせ
    df['EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCES_WEIGHTED'] = (df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 4) / 9
    # クレジットと商品価格の関係
    df['DOWN_PAYMENT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['DOWN_PAYMENT_RATIO'] = df['DOWN_PAYMENT'] / df['AMT_GOODS_PRICE']


    # 6rd の特徴量を追加
    ### 四則演算で作る特徴量
    # DAYS系
    df["YEAR_EMPLOYED"] = df["DAYS_EMPLOYED"]/365
    df["YEAR_BIRTH"] = df["DAYS_BIRTH"]/365
    df["YEAR_REGISTRATION"] = df["DAYS_REGISTRATION"]/365
    df["YEAR_ID_PUBLISH"] = df["DAYS_ID_PUBLISH"]/365

    df["YEAR_ID_PUBLISH_2"] = df["YEAR_ID_PUBLISH"]*df["YEAR_ID_PUBLISH"]
    df["YEAR_REGISTRATION_2"] = df["YEAR_REGISTRATION"]*df["YEAR_REGISTRATION"]
    df["YEAR_BIRTH_2"] = df["YEAR_BIRTH"]*df["YEAR_BIRTH"]
    df["YEAR_EMPLOYED_2"] = df["YEAR_EMPLOYED"]*df["YEAR_EMPLOYED"]

    df["ID_PUBLISH_REGISTRATION_hiku"] = df["YEAR_REGISTRATION"] - df["YEAR_ID_PUBLISH"]

    df["EMPLOYED_BIRTH_mainas"] = df["DAYS_BIRTH"] - df["DAYS_EMPLOYED"]
    df["EMPLOYED_BIRTH_waru"] = df["YEAR_EMPLOYED"]/df["YEAR_BIRTH"]
    df["REGISTRATION_BIRTH"] = df["YEAR_REGISTRATION"]/df["YEAR_BIRTH"]
    df["ID_PUBLISH_BIRTH"] = df["YEAR_ID_PUBLISH"]/df["YEAR_BIRTH"]
    df["ID_PUBLISH_REGISTRATION_kake"] = df["YEAR_REGISTRATION"]*df["YEAR_ID_PUBLISH"]


    # AMT系
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL']/df['AMT_CREDIT']
    df['GOODS_PERC'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT']
    df['RATIO_INCOME_GOODS_hiku'] = df['AMT_INCOME_TOTAL'] -  df['AMT_GOODS_PRICE']
    df['ANNUITY_LENGTH_CR'] = df['AMT_CREDIT']/df['AMT_ANNUITY']# 借入額と毎月の支払額の比
    df['ANNUITY_LENGTH_CR_2'] = (df['AMT_CREDIT']/df['AMT_ANNUITY'])**2# 借入額と毎月の支払額の比

    df["ANNUITY_LENGTH_BIRTH"] = df["YEAR_BIRTH"] + df['ANNUITY_LENGTH_CR']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_INCOME_TOTAL']/df['AMT_ANNUITY']

    df['CREDIT_YEAR_ID'] = df['AMT_CREDIT']/df["YEAR_ID_PUBLISH"]
    df['CREDIT_YEAR_REGISTRATION'] = df['AMT_CREDIT']/df["YEAR_REGISTRATION"]
    df["ANNUITY_LENGTH_ID_PUBLISH"] = df['ANNUITY_LENGTH_CR']/df["YEAR_ID_PUBLISH"]


    # 金利
    df["kinri"] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']

    # 一人当たりで考える
    df['INCOME_PER_FAMILY_MEMBER'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']

    # EXT系
    df['app EXT_SOURCE mean'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
    df['app EXT_SOURCE std'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
    df["app EXT_SOURCE max"] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis = 1)
    df["app EXT_SOURCE min"] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis = 1)
    df['app EXT_SOURCE max_1'] = df['app EXT_SOURCE max'] -  df['EXT_SOURCE_1']
    df['app EXT_SOURCE max_2'] = df['app EXT_SOURCE max'] -  df['EXT_SOURCE_2']
    df['app EXT_SOURCE max_3'] = df['app EXT_SOURCE max'] -  df['EXT_SOURCE_3']
    df['app EXT_SOURCE min_1'] = df['app EXT_SOURCE min'] -  df['EXT_SOURCE_1']
    df['app EXT_SOURCE min_2'] = df['app EXT_SOURCE min'] -  df['EXT_SOURCE_2']
    df['app EXT_SOURCE min_3'] = df['app EXT_SOURCE min'] -  df['EXT_SOURCE_3']

    df['app EXT_SOURCE_1_EXT_SOURCE_2_kake'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['app EXT_SOURCE_1_EXT_SOURCE_3_kake'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['app EXT_SOURCE_2_EXT_SOURCE_3_kake'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # EXT＋α系

    df['app EXT_SOURCE_1_DAYS_BIRTH_waru'] = df['EXT_SOURCE_1'] / df['YEAR_BIRTH']
    df['app EXT_SOURCE_2_DAYS_BIRTH_waru'] = df['EXT_SOURCE_2'] / df['YEAR_BIRTH']
    df['app EXT_SOURCE_3_DAYS_BIRTH_waru'] = df['EXT_SOURCE_3'] / df['YEAR_BIRTH']

    df.loc[df["DAYS_EMPLOYED"]==0, "DAYS_EMPLOYED"] = 0.01
    df['app EXT_SOURCE_1_DAYS_EMPLOYED_waru'] = df['EXT_SOURCE_1'] / df['YEAR_EMPLOYED']
    df['app EXT_SOURCE_2_DAYS_EMPLOYED_waru'] = df['EXT_SOURCE_2'] / df['YEAR_EMPLOYED']
    df['app EXT_SOURCE_3_DAYS_EMPLOYED_waru'] = df['EXT_SOURCE_3'] / df['YEAR_EMPLOYED']
    df.loc[df["DAYS_EMPLOYED"]==0.01, "DAYS_EMPLOYED"] = 0


    # FICOスコアを再現する
    df["EXT_SOURCE_1_CUT"] = pd.cut(df["EXT_SOURCE_1"], 550, labels=False)
    df["EXT_SOURCE_2_CUT"] = pd.cut(df["EXT_SOURCE_2"], 550, labels=False)
    df["EXT_SOURCE_3_CUT"] = pd.cut(df["EXT_SOURCE_3"], 550, labels=False)
    df.loc[df['EXT_SOURCE_1_CUT'] <= 259, 'EXT_SOURCE_1_CUT'] = 0
    df.loc[(df['EXT_SOURCE_1_CUT'] > 260) & (df['EXT_SOURCE_1_CUT'] <= 359), 'EXT_SOURCE_1_CUT'] = 1
    df.loc[(df['EXT_SOURCE_1_CUT'] > 360) & (df['EXT_SOURCE_1_CUT'] <= 424), 'EXT_SOURCE_1_CUT'] = 2
    df.loc[(df['EXT_SOURCE_1_CUT'] > 425) & (df['EXT_SOURCE_1_CUT'] <= 459), 'EXT_SOURCE_1_CUT'] = 3
    df.loc[df['EXT_SOURCE_1_CUT'] > 459, 'EXT_SOURCE_1_CUT'] = 4

    df.loc[df['EXT_SOURCE_2_CUT'] <= 259, 'EXT_SOURCE_2_CUT'] = 0
    df.loc[(df['EXT_SOURCE_2_CUT'] > 260) & (df['EXT_SOURCE_2_CUT'] <= 359), 'EXT_SOURCE_2_CUT'] = 1
    df.loc[(df['EXT_SOURCE_2_CUT'] > 360) & (df['EXT_SOURCE_2_CUT'] <= 424), 'EXT_SOURCE_2_CUT'] = 2
    df.loc[(df['EXT_SOURCE_2_CUT'] > 425) & (df['EXT_SOURCE_2_CUT'] <= 459), 'EXT_SOURCE_2_CUT'] = 3
    df.loc[df['EXT_SOURCE_2_CUT'] > 459, 'EXT_SOURCE_2_CUT'] = 4

    df.loc[df['EXT_SOURCE_3_CUT'] <= 259, 'EXT_SOURCE_3_CUT'] = 0
    df.loc[(df['EXT_SOURCE_3_CUT'] > 260) & (df['EXT_SOURCE_3_CUT'] <= 359), 'EXT_SOURCE_3_CUT'] = 1
    df.loc[(df['EXT_SOURCE_3_CUT'] > 360) & (df['EXT_SOURCE_3_CUT'] <= 424), 'EXT_SOURCE_3_CUT'] = 2
    df.loc[(df['EXT_SOURCE_3_CUT'] > 425) & (df['EXT_SOURCE_3_CUT'] <= 459), 'EXT_SOURCE_3_CUT'] = 3
    df.loc[df['EXT_SOURCE_3_CUT'] > 459, 'EXT_SOURCE_3_CUT'] = 4
    df['EXT_SOURCE_CUT_MEAN'] = df[['EXT_SOURCE_1_CUT', 'EXT_SOURCE_2_CUT', 'EXT_SOURCE_3_CUT']].mean(axis = 1)

    # socialcircle_feature,bureau_featureの二値化
    df["NO_DEF_30L_CIRCLE"] = 0
    df.loc[df["DEF_30_CNT_SOCIAL_CIRCLE"]>=1, 'NO_DEF_30L_CIRCLE'] = 1
    df["NO_DEF_60L_CIRCLE"] = 0
    df.loc[df["DEF_60_CNT_SOCIAL_CIRCLE"]>=1, 'NO_DEF_60L_CIRCLE'] = 1
    df["NO_OBS_30L_CIRCLE"] = 0
    df.loc[df["OBS_30_CNT_SOCIAL_CIRCLE"]>=1, 'NO_OBS_30L_CIRCLE'] = 1
    df["NO_OBS_60L_CIRCLE"] = 0
    df.loc[df["OBS_60_CNT_SOCIAL_CIRCLE"]>=1, 'NO_OBS_60L_CIRCLE'] = 1
    df["NO_BUREAU_HOUR"] = 0
    df.loc[df["AMT_REQ_CREDIT_BUREAU_HOUR"]>=1, 'NO_BUREAU_HOUR'] = 1
    df["NO_BUREAU_MON"] = 0
    df.loc[df["AMT_REQ_CREDIT_BUREAU_MON"]>=1, 'NO_BUREAU_MON'] = 1
    df["NO_BUREAU_QRT"] = 0
    df.loc[df["AMT_REQ_CREDIT_BUREAU_QRT"]>=1, 'NO_BUREAU_QRT'] = 1
    df["NO_BUREAU_YEAR"] = 0
    df.loc[df["AMT_REQ_CREDIT_BUREAU_YEAR"]>=1, 'NO_BUREAU_YEAR'] = 1

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
    
    # データフレームにすでにneighbors_target_mean_500列が存在する場合は削除
    if 'neighbors_target_mean_500' in df.columns:
        df = df.drop('neighbors_target_mean_500', axis=1)
    
    # データフレームにneighbors_target_mean_500を追加
    df = pd.merge(df, neighbors_target_mean_500_df, on='SK_ID_CURR', how='left')
    
    return df

def to_drop(df):
    
    drop_list = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'REG_REGION_NOT_LIVE_REGION', 'LIVE_REGION_NOT_WORK_REGION']
    droped_df = df.drop(columns=drop_list)
    
    return droped_df

lgbm_best_param = {
    'reg_lambda': 1.1564659040946654,
    'reg_alpha': 9.90877329623665, 
    'colsample_bytree': 0.5034991685866442, 
    'subsample': 0.6055998601661783, 
    'max_depth': 3, 
    'min_child_weight': 39.72586351155486, 
    'learning_rate': 0.08532489659779158, 
    'verbose': -1,
    'device': 'gpu',
    'gpu_platform_id': 0,
    'gpu_device_id': 0,
    'objective': 'binary',
    'boost_from_average': True,
    'force_col_wise': True
}

def fit_lgbm(X, y, cv, params: dict=None, verbose=100):
    oof_preds = np.zeros(X.shape[0])
    if params is None: params = {}
    models = []
    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]

        clf = lgbm.LGBMClassifier(**params, random_state=71, n_estimators=10000)
        clf.fit(x_train, y_train, 
                eval_set=[(x_valid, y_valid)],  
                eval_metric='auc',
                #early_stopping_rounds=100, 
                #verbose=verbose
                )

        models.append(clf)
        oof_preds[idx_valid] = clf.predict_proba(x_valid, num_iteration=clf.best_iteration_)[:, 1]
        print('Fold %2d AUC : %.6f' % (i + 1, roc_auc_score(y_valid, oof_preds[idx_valid])))
    
    score = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % score) 
    return oof_preds, models

def check_gpu_availability():
    try:
        lgbm.Dataset(None).construct()
    except lgbm.basic.LightGBMError as err:
        print(f"LightGBM Error: {err}")
    else:
        print("GPU is available for LightGBM.")

def fit_lgbm_gpu(X, y, cv, params: dict=None, verbose=100):
    oof_preds = np.zeros(X.shape[0])
    if params is None: params = {}
    models = []
    for i, (idx_train, idx_valid) in enumerate(cv):
        x_train, y_train = X[idx_train], y[idx_train]
        x_valid, y_valid = X[idx_valid], y[idx_valid]
        
        train_dataset = lgbm.Dataset(x_train, label=y_train)
        valid_dataset = lgbm.Dataset(x_valid, label=y_valid)
        
        # Ensure GPU usage
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        
        clf = lgbm.train(
            params, 
            train_dataset,
            num_boost_round=10000,
            valid_sets=[train_dataset, valid_dataset],
            callbacks=[lgbm.early_stopping(100), lgbm.log_evaluation(verbose)]
        )
        
        models.append(clf)
        oof_preds[idx_valid] = clf.predict(x_valid)
        print('Fold %2d AUC : %.6f' % (i + 1, roc_auc_score(y_valid, oof_preds[idx_valid])))
    
    score = roc_auc_score(y, oof_preds)
    print('Full AUC score %.6f' % score) 
    return oof_preds, models

# ログファイルを初期化
with open("C:/Gdrive/data2/results.log", "w") as f:
    pass

# 標準出力をログファイルにリダイレクト（追記）しながら、ターミナルにも出力する
sys.stdout = DualLogger("C:/Gdrive/data2/results.log", sys.stdout)

print_with_timestamp('データを読み込みます...')
train_df = pd.read_csv('C:/Gdrive/data2/train.csv')
test_df = pd.read_csv('C:/Gdrive/data2/test.csv')
sample_submission = pd.read_csv('C:/Gdrive/data2/sample_submission.csv')
df_all = pd.concat([train_df, test_df], axis=0)
print_with_timestamp('データ読み込み完了')

print_with_timestamp('dfをone-hot encodingします...')
df_encoded = one_hot_encoding(df_all)
print_with_timestamp('one-hot encoding完了')

print_with_timestamp('主要な特徴量を追加します...')
added_features_df = to_add_feature(df_encoded)
print_with_timestamp('主要な特徴量追加完了')

#新規特徴量が無い時は、ここ3行はコメントアウト可能（計算が早くなる）
# print_with_timestamp('neighbors_target_mean_500を計算して追加します...') 
# added_features_df = add_features_parallel(added_features_df) 
# print_with_timestamp('neighbors_target_mean_500計算追加完了') 

print_with_timestamp('neighbors_target_mean_500を外部ファイルから追加します...')
added_features_df = add_neighbors_target_mean_500(added_features_df)
print_with_timestamp('neighbors_target_mean_500追加完了')

print_with_timestamp('不要な特徴量を削除します...')
all_features_df = to_drop(added_features_df)
print_with_timestamp('不要な特徴量削除完了')

# データの整合性を確認
assert len(df_all) == len(df_encoded)
assert len(df_all) == len(added_features_df)
assert len(df_all) == len(all_features_df)

# データをtrainとtestに分割
train = all_features_df[all_features_df.loc[:, 'SK_ID_CURR'] < 171202]
test = all_features_df[all_features_df.loc[:, 'SK_ID_CURR'] > 171201]

# データをCSVファイルに保存
print_with_timestamp('前処理済みデータをCSVファイルに保存します...')
train.to_csv('C:/Gdrive/data2/train_processed.csv', index=False)
test.to_csv('C:/Gdrive/data2/test_processed.csv', index=False)
print_with_timestamp('保存完了')

train_x = train.drop(columns=['TARGET', 'SK_ID_CURR'])
train_y = train['TARGET']
test_x = test.drop(columns=['TARGET', 'SK_ID_CURR'])

X = train_x.values
y = train_y.values

fold = StratifiedKFold(n_splits=8, shuffle=True, random_state=69)
cv = list(fold.split(X, y))

# メインの予測部分
print_with_timestamp('LightGBMで学習を行います...')
check_gpu_availability()
oof, models = fit_lgbm_gpu(X, y, cv=cv, params=lgbm_best_param)
print_with_timestamp('学習完了')

pred = np.array([model.predict(test_x.values) for model in models])  # Changed from predict_proba to predict
pred = np.mean(pred, axis=0)

submission = sample_submission.copy()
submission['TARGET'] = pred

print_with_timestamp('予測結果を出力します...')
submission.to_csv('C:/Gdrive/data2/3rd_place_solution.csv', index=False)
print_with_timestamp('出力完了')