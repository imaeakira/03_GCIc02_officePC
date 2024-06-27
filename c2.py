import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgbm
from lightgbm import early_stopping
from lightgbm import log_evaluation

from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import VarianceThreshold
import os
from category_encoders import CatBoostEncoder, WOEEncoder
import category_encoders as ce
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fancyimpute import SoftImpute
import multiprocessing
from numba import jit
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import combinations
from tqdm import tqdm

import datetime
import time
last_time = None  # 前回の時間を保持するグローバル変数
chunk_size = 500000  # 開始値を設定

# 前処理をスキップするかどうかのフラグ
SKIP_PREPROCESSING = False  # Trueに変更すると前処理をスキップします

# neighbors_target_mean_* 列の最適な組み合わせ探索を行うかどうかのフラグ
OPTIMIZE_NEIGHBORS_COLUMNS = False  # Trueに変更すると組み合わせ探索を行います

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

def load_preprocessed_data():
    print_with_timestamp('前処理済みデータを読み込みます...')
    train = pd.read_csv('C:/Gdrive/data2/train_processed.csv')
    test = pd.read_csv('C:/Gdrive/data2/test_processed.csv')
    sample_submission = pd.read_csv('C:/Gdrive/data2/sample_submission.csv')
    print_with_timestamp('前処理済みデータの読み込み完了')
    return train, test

def to_drop(df):
    print_with_timestamp('不要な特徴量を削除します...')

    # drop_list = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'REG_REGION_NOT_LIVE_REGION', 'LIVE_REGION_NOT_WORK_REGION']
    drop_list = ['FLAG_MOBIL'] # trainで1、testで0の特徴量なので完全に不要、残りは少しでも情報があるので残す

    droped_df = df.drop(columns=drop_list)
    
    print_with_timestamp('不要な特徴量の削除完了')
    return droped_df

def to_add_feature(df):
    print_with_timestamp('主要な特徴量を追加します...')

    # region GCI2020Summer3rd を参考に追加
    # 異常値を欠損値に（365243 -> np.nan）
    # df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan) #2nd solutionで実施済
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
    # endregion

    # region Kaggle 1st solution を参考に追加
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
    # endregion

    # region GCI2020Summer6th を参考に追加
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
    # endregion

    # region Kaggle 2nd solution を参考に追加
    ## Yamamoto-sanのsolution文章＆000.pyからClaude3.5sonnetが抽出した特徴量
    # 金利（Interest rate）の計算
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['INTEREST_RATE'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY']) ** (1 / df['CREDIT_TERM']) - 1
    # 時間経過に伴う金利の変化
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # EXT_SOURCE_Xを使用したターゲットエンコーディング
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # 追加の金利関連特徴量 nejumiという特殊な金利計算方法
    f_name = 'nejumi'
    init_rate = 0.9
    n_iter = 500
    # CNT_PAYMENTが存在しない場合、推定を試みる
    if 'CNT_PAYMENT' not in df.columns:
        if 'AMT_CREDIT' in df.columns and 'AMT_ANNUITY' in df.columns:
            df['CNT_PAYMENT'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        else:
            print("Warning: Unable to calculate or estimate CNT_PAYMENT. Skipping nejumi feature.")
            return df
    df['AMT_ANNUITY_d_AMT_CREDIT_temp'] = df.AMT_ANNUITY / df.AMT_CREDIT   
    df[f_name] = df['AMT_ANNUITY_d_AMT_CREDIT_temp'] * ((1 + init_rate)**df.CNT_PAYMENT - 1) / ((1 + init_rate)**df.CNT_PAYMENT)
    for _ in range(n_iter):
        df[f_name] = df['AMT_ANNUITY_d_AMT_CREDIT_temp'] * ((1 + df[f_name])**df.CNT_PAYMENT - 1) / ((1 + df[f_name])**df.CNT_PAYMENT) 
    df.drop(['AMT_ANNUITY_d_AMT_CREDIT_temp'], axis=1, inplace=True)
    # EXT_SOURCE関連の追加特徴量
    df['EXT_SOURCE_1_2_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_2']) / 2
    df['EXT_SOURCE_1_3_mean'] = (df['EXT_SOURCE_1'] + df['EXT_SOURCE_3']) / 2
    df['EXT_SOURCE_2_3_mean'] = (df['EXT_SOURCE_2'] + df['EXT_SOURCE_3']) / 2
    df['EXT_SOURCE_1_2_diff'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_2'])
    df['EXT_SOURCE_1_3_diff'] = abs(df['EXT_SOURCE_1'] - df['EXT_SOURCE_3'])
    df['EXT_SOURCE_2_3_diff'] = abs(df['EXT_SOURCE_2'] - df['EXT_SOURCE_3'])
    df['EXT_SOURCE_1_2_3_weighted'] = (df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 4) / 9
    # 申請金額と信用限度額の関係
    df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['DOWN_PAYMENT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['DOWN_PAYMENT_RATIO'] = df['DOWN_PAYMENT'] / df['AMT_GOODS_PRICE']
    # 時間関連の特徴量
    df['CREDIT_TO_BIRTH_RATIO'] = df['AMT_CREDIT'] / (-df['DAYS_BIRTH'])
    df['CREDIT_TO_EMPLOYED_RATIO'] = df['AMT_CREDIT'] / (-df['DAYS_EMPLOYED'])
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / (-df['DAYS_EMPLOYED'])
    df['INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / (-df['DAYS_BIRTH'])
    # 家族構成関連の特徴量
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_PER_PERSON'] = df['AMT_ANNUITY'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    # ドキュメント関連の特徴量
    docs = [f for f in df.columns if 'FLAG_DOC' in f]
    df['DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    df['DOCUMENT_KURTOSIS'] = df[docs].kurtosis(axis=1)
    df['DOCUMENT_SKEW'] = df[docs].skew(axis=1)
    ## Githubの000.pyからCalude3.5sonnetが改めて抽出＆再現した特徴量
    # 信用関連の比率
    # クレジット額と年収の比率
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    # 年間支払額と年収の比率
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # クレジット期間（年）
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # 雇用期間と年齢の比率
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    # 退職者フラグ（雇用日数が365243の場合は退職者と見なす）
    df['DAYS_EMPLOYED_ANOMALY'] = df['DAYS_EMPLOYED'] == 365243
    df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    # 無収入フラグ
    df['NO_INCOME_FLAG'] = (df['AMT_INCOME_TOTAL'] == 0).astype(int)
    # 家族1人当たりの収入
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    # 年齢（年）
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH'] // 365)
    # # カテゴリカル変数のlabel encoding
    # categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    # le = LabelEncoder()
    # for col in categorical_columns:
    #     df[col] = le.fit_transform(df[col].astype(str))
    # FLAG_ で始まる列を処理
    flag_columns = [col for col in df.columns if 'FLAG_' in col]
    numeric_flag_columns = []
    for col in flag_columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            numeric_flag_columns.append(col)
        except ValueError:
            print(f"Warning: Column {col} could not be converted to numeric. It will be excluded from TOTAL_FLAGS calculation.")
    # TOTAL_FLAGS の計算（数値に変換できた列のみを使用）
    df['TOTAL_FLAGS'] = df[numeric_flag_columns].sum(axis=1)
    # 提出書類の数
    df['TOTAL_DOCUMENTS'] = df[[col for col in df.columns if 'FLAG_DOCUMENT' in col]].sum(axis=1)
    # EXT_SOURCE特徴量
    df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    # クレジット額と商品価格の差額
    df['CREDIT_GOODS_PRICE_DIFF'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    # 収入タイプごとの収入の中央値に対する比率
    income_by_type = df[['AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE']].groupby('NAME_INCOME_TYPE').median()['AMT_INCOME_TOTAL']
    df['INCOME_TYPE_RATIO'] = df['AMT_INCOME_TOTAL'] / df['NAME_INCOME_TYPE'].map(income_by_type)
    # クレジット額と商品価格の比率
    df['CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # 頭金とその比率
    df['DOWN_PAYMENT'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['DOWN_PAYMENT_RATIO'] = df['DOWN_PAYMENT'] / df['AMT_GOODS_PRICE']
    # 地域評価に関する特徴量
    df['REGION_RATING_CLIENT_W_CITY_MAX'] = df[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].max(axis=1)
    df['REGION_RATING_CLIENT_W_CITY_DIFF'] = df['REGION_RATING_CLIENT_W_CITY'] - df['REGION_RATING_CLIENT']
    # 電話変更に関する特徴量
    if 'DAYS_LAST_PHONE_CHANGE' in df.columns:
        # 0をNaNに置き換え
        df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan)
        
        # PHONE_CHANGE_YEAR の計算
        # NaNと無限大の値を保持したまま計算し、それ以外の値のみ整数に変換
        df['PHONE_CHANGE_YEAR'] = (abs(df['DAYS_LAST_PHONE_CHANGE']) // 365)
        mask = df['PHONE_CHANGE_YEAR'].notna() & np.isfinite(df['PHONE_CHANGE_YEAR'])
        df.loc[mask, 'PHONE_CHANGE_YEAR'] = df.loc[mask, 'PHONE_CHANGE_YEAR'].astype(int)
    else:
        print("Warning: DAYS_LAST_PHONE_CHANGE column is missing. Skipping PHONE_CHANGE_YEAR calculation.")
    # 家族状況に関する特徴量
    df['NAME_FAMILY_STATUS_MARRIED'] = (df['NAME_FAMILY_STATUS'] == 'Married').astype(int)
    df['NAME_FAMILY_STATUS_SINGLE'] = (df['NAME_FAMILY_STATUS'] == 'Single / not married').astype(int)
    df['NAME_EDUCATION_TYPE_HIGHER'] = (df['NAME_EDUCATION_TYPE'] == 'Higher education').astype(int)
    # 社会的サークルにおけるデフォルト率
    df['SOCIAL_CIRCLE_DEFAULT_RATIO'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + 1)
    df['SOCIAL_CIRCLE_DEFAULT_60_RATIO'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / (df['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)
    # 雇用と年齢の関係
    df['EMPLOYMENT_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['EMPLOYMENT_START_AGE'] = (df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']) / 365
    # ドキュメントの提出状況
    document_columns = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
    df['DOCUMENT_SUBMISSION_RATIO'] = df[document_columns].sum(axis=1) / len(document_columns)
    # 職業と収入の関係
    df['INCOME_TO_ORGANIZATION_MEAN_RATIO'] = df['AMT_INCOME_TOTAL'] / df.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
    # 家族構成と収入の関係
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    # 資産と負債の比率
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['GOODS_PRICE_TO_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    # 時間関連の特徴量
    df['REGISTRATION_TO_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_REGISTRATION']
    df['ID_PUBLISH_TO_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'] - df['DAYS_ID_PUBLISH']
    # endregion

    # region tsfresh を使いたかったがだめだったので、手動で時系列特徴量を追加
    time_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
    # 絶対値を取る
    for col in time_cols:
        df[f'{col}_ABS'] = np.abs(df[col])
    # 統計量を計算
    df['DAYS_MEAN'] = df[time_cols].mean(axis=1)
    df['DAYS_STD'] = df[time_cols].std(axis=1)
    df['DAYS_MIN'] = df[time_cols].min(axis=1)
    df['DAYS_MAX'] = df[time_cols].max(axis=1)
    # 比率を計算
    for col in time_cols:
        df[f'{col}_TO_BIRTH_RATIO'] = df[col] / df['DAYS_BIRTH']
    # 差分を計算
    for i in range(len(time_cols)):
        for j in range(i+1, len(time_cols)):
            df[f'{time_cols[i]}_MINUS_{time_cols[j]}'] = df[time_cols[i]] - df[time_cols[j]]
    # 追加の特徴量
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    # endregion

    print_with_timestamp('主要な特徴量の追加完了')
    return df

def add_leak_features(df_all):
    print_with_timestamp('同一ユーザー特徴量の追加を開始します...')
    
    # ユーザー識別のための特徴量を作成
    def create_user_features(df):
        df['DAYS_BIRTH-m-DAYS_REGISTRATION'] = df['DAYS_BIRTH'] - df['DAYS_REGISTRATION']
        df['DAYS_REGISTRATION-m-DAYS_ID_PUBLISH'] = df['DAYS_REGISTRATION'] - df['DAYS_ID_PUBLISH']
        df['DAYS_ID_PUBLISH-m-DAYS_EMPLOYED'] = df['DAYS_ID_PUBLISH'] - df['DAYS_EMPLOYED']
        return df

    df_all = create_user_features(df_all)

    # ユーザー識別のためのキー
    keys = ['DAYS_BIRTH-m-DAYS_REGISTRATION', 'DAYS_REGISTRATION-m-DAYS_ID_PUBLISH',
            'DAYS_ID_PUBLISH-m-DAYS_EMPLOYED', 'CODE_GENDER', 'NAME_EDUCATION_TYPE']

    # グループ化
    grouped = df_all.groupby(keys)

    # 1. user_credit_score_change
    df_all['user_credit_score_change'] = grouped['EXT_SOURCE_1'].transform(lambda x: x.max() - x.min())

    # 2. user_payment_history_improvement
    df_all['user_payment_history_improvement'] = grouped['SK_ID_CURR'].transform(lambda x: x.count() - 1)

    # 3. user_debt_to_income_ratio_change
    df_all['debt_to_income_ratio'] = df_all['AMT_CREDIT'] / df_all['AMT_INCOME_TOTAL']
    df_all['user_debt_to_income_ratio_change'] = grouped['debt_to_income_ratio'].transform(lambda x: x.max() - x.min())

    # 4. user_credit_amount_change
    df_all['user_credit_amount_change'] = grouped['AMT_CREDIT'].transform(lambda x: (x.max() - x.min()) / x.min())

    # 5. user_income_change
    df_all['user_income_change'] = grouped['AMT_INCOME_TOTAL'].transform(lambda x: (x.max() - x.min()) / x.min())

    # 6. user_credit_utilization_change
    df_all['credit_utilization'] = df_all['AMT_CREDIT'] / df_all['AMT_GOODS_PRICE']
    df_all['user_credit_utilization_change'] = grouped['credit_utilization'].transform(lambda x: x.max() - x.min())

    # 7. user_max_concurrent_loans
    df_all['user_max_concurrent_loans'] = grouped['SK_ID_CURR'].transform('count')

    # 8. user_credit_limit_utilization_trend
    df_all['credit_limit_utilization'] = df_all['AMT_CREDIT'] / grouped['AMT_CREDIT'].transform('max')
    df_all['user_credit_limit_utilization_trend'] = grouped['credit_limit_utilization'].transform(lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0)

    # 9. user_application_frequency
    df_all['user_application_frequency'] = grouped['SK_ID_CURR'].transform('count') / (grouped['DAYS_REGISTRATION'].transform('max') - grouped['DAYS_REGISTRATION'].transform('min')).abs()

    # 10. user_days_difference
    df_all['user_days_difference'] = grouped['DAYS_REGISTRATION'].transform(lambda x: x.max() - x.min())

    # 11. user_previous_applications_count
    df_all['user_previous_applications_count'] = grouped['SK_ID_CURR'].transform('cumcount')

    # 12. user_employment_stability
    df_all['user_employment_stability'] = grouped['ORGANIZATION_TYPE'].transform('nunique')

    # 13. user_collateral_value_change
    df_all['user_collateral_value_change'] = grouped['AMT_GOODS_PRICE'].transform(lambda x: (x.max() - x.min()) / x.min())

    # 14. dup_count
    df_all['dup_count'] = grouped.cumcount() + 1

    # 15. user_avg_time_between_applications
    df_all['user_avg_time_between_applications'] = grouped['DAYS_REGISTRATION'].transform(lambda x: (x.max() - x.min()) / (len(x) - 1) if len(x) > 1 else 0)

    # 16. user_loan_purpose_diversity
    le = LabelEncoder()
    df_all['NAME_CONTRACT_TYPE_encoded'] = le.fit_transform(df_all['NAME_CONTRACT_TYPE'])
    df_all['user_loan_purpose_diversity'] = grouped['NAME_CONTRACT_TYPE_encoded'].transform('nunique')

    # 17. user_employment_duration_change
    df_all['user_employment_duration_change'] = grouped['DAYS_EMPLOYED'].transform(lambda x: x.max() - x.min())

    # 18. user_max_credit_amount
    df_all['user_max_credit_amount'] = grouped['AMT_CREDIT'].transform('max')

    # 19. user_seasonal_application_pattern
    df_all['application_month'] = (-df_all['DAYS_REGISTRATION'] % 365 // 30 + 1).astype(int)  # 概算の月を計算
    df_all['user_seasonal_application_pattern'] = grouped['application_month'].transform('nunique')

    # 20. user_min_credit_amount
    df_all['user_min_credit_amount'] = grouped['AMT_CREDIT'].transform('min')

    # 訓練データとテストデータを分割
    train_df = df_all[df_all['TARGET'].notnull()].copy()
    test_df = df_all[df_all['TARGET'].isnull()].copy()

    # 訓練データ内で重複するユーザーを特定
    train_duplicates = train_df[train_df.duplicated(keys, keep=False)]
    
    # 重複ユーザーにIDを割り当て（一時的な列として）
    train_duplicates['temp_user_id'] = train_duplicates.groupby(keys).ngroup()
    
    # ユーザーごとのTARGET平均を計算
    user_target_mean = train_duplicates.groupby('temp_user_id')['TARGET'].mean().reset_index()
    user_target_mean.columns = ['temp_user_id', 'user_target_mean']
    
    # 訓練データの重複ユーザーにuser_target_meanを追加
    train_df = train_df.merge(train_duplicates[['SK_ID_CURR', 'temp_user_id']], on='SK_ID_CURR', how='left')
    train_df = train_df.merge(user_target_mean, on='temp_user_id', how='left')
    
    # テストデータ内のユーザーを訓練データとマッチング
    test_df = test_df.merge(train_duplicates[keys + ['temp_user_id']].drop_duplicates(), on=keys, how='left')
    
    # テストデータの該当ユーザーにuser_target_meanを追加
    test_df = test_df.merge(user_target_mean, on='temp_user_id', how='left')

    # 一時的なuser_id列を削除
    train_df = train_df.drop('temp_user_id', axis=1)
    test_df = test_df.drop('temp_user_id', axis=1)

    # 訓練データとテストデータを再結合
    df_all = pd.concat([train_df, test_df], axis=0).sort_index()

    print_with_timestamp('同一ユーザー特徴量の追加が完了しました')
    
    # デバッグ情報の出力
    print(f"訓練データでuser_target_meanが非NULLの数: {train_df['user_target_mean'].notnull().sum()}")
    print(f"テストデータでuser_target_meanが非NULLの数: {test_df['user_target_mean'].notnull().sum()}")
    
    return df_all

def one_hot_encoding(df):
    return pd.get_dummies(df, drop_first=False)

def label_encoding(df, cat_columns):
    le = LabelEncoder()
    df_encoded = df.copy()
    
    for col in cat_columns:
        df_encoded[f'{col}_label'] = le.fit_transform(df[col].astype(str))
    
    return df_encoded

def apply_all_encodings(df, keep_original=False):
    print_with_timestamp('カテゴリ変数をone-hot, label encodingします...')
    
    # カテゴリカル変数を特定
    cat_columns = df.select_dtypes(include=['object']).columns
    non_cat_columns = [col for col in df.columns if col not in cat_columns and col not in ['TARGET', 'SK_ID_CURR']]
    
    # 欠損値を 'Unknown' で埋める
    for col in cat_columns:
        df[col] = df[col].fillna('Unknown')
    
    # One-hot encoding
    df_onehot = one_hot_encoding(df[cat_columns])
    
    # Label encoding
    df_label = label_encoding(df, cat_columns)
    
    # 全てのエンコーディング結果を結合
    df_encoded = pd.concat([
        df_onehot,
        # df_label[[col for col in df_label.columns if col.endswith('_label')]],
        df[non_cat_columns],
        df[['TARGET', 'SK_ID_CURR']]
    ], axis=1)
    
    if keep_original:
        # 元のカテゴリカル変数を保持
        df_encoded = pd.concat([df_encoded, df[cat_columns]], axis=1)
    
    print_with_timestamp('カテゴリ変数のone-hot, label encoding完了')
    # print(f"エンコーディング前の列数: {df.shape[1]}, エンコーディング後の列数: {df_encoded.shape[1]}")
    # print(f"保持されたカテゴリカル変数: {cat_columns.tolist()}")
    return df_encoded

def improved_target_encode_features(df, cat_columns, target_column='TARGET', n_splits=5, shuffle=True, random_state=69):
    print_with_timestamp('カテゴリ変数のtarget encodingを開始します...')
    
    # 訓練データとテストデータを分離
    train_df = df[df[target_column].notna()].copy()
    test_df = df[df[target_column].isna()].copy()
    
    # 交差検証の設定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    # 訓練データのターゲットエンコーディング
    train_encoded = train_df.copy()
    for col in cat_columns:
        train_encoded[f'{col}_target_encoded'] = 0
    
    te = TargetEncoder()
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[target_column])):
        print(f'Processing fold {fold + 1}/{n_splits}')
        X_train, X_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
        
        te.fit(X_train[cat_columns], X_train[target_column])
        
        encoded_cols = [f'{col}_target_encoded' for col in cat_columns]
        train_encoded.loc[val_idx, encoded_cols] = te.transform(X_val[cat_columns])
    
    # テストデータのターゲットエンコーディング
    te_full = TargetEncoder()
    te_full.fit(train_df[cat_columns], train_df[target_column])
    
    test_encoded = test_df.copy()
    encoded_cols = [f'{col}_target_encoded' for col in cat_columns]
    test_encoded[encoded_cols] = te_full.transform(test_df[cat_columns])
    
    # 訓練データとテストデータを結合
    df_encoded = pd.concat([train_encoded, test_encoded], axis=0)
    
    print_with_timestamp('カテゴリ変数のtarget encodingが完了しました')
    return df_encoded

def improved_categorical_encoding(df):
    print_with_timestamp('カテゴリ変数をCatBoost, WOEエンコーディングします...')
    
    # カテゴリカル変数を特定（ただし、TARGET列は除外）
    cat_columns = df.select_dtypes(include=['object']).columns
    target_column = 'TARGET'
    
    # トレーニングデータとテストデータを分割
    train_df = df[df[target_column].notnull()].copy()
    test_df = df[df[target_column].isnull()].copy()

    # CatBoost エンコーディング
    ce_encoder = CatBoostEncoder(cols=cat_columns, random_state=42)
    ce_encoder.fit(train_df[cat_columns], train_df[target_column])

    # エンコーディングを適用
    train_df_encoded = ce_encoder.transform(train_df[cat_columns])
    test_df_encoded = ce_encoder.transform(test_df[cat_columns])

    # 元のデータフレームに新しい特徴量を追加
    for col in cat_columns:
        new_col_name = f'{col}_catboost'
        train_df[new_col_name] = train_df_encoded[col]
        test_df[new_col_name] = test_df_encoded[col]

    # WOE エンコーディング
    woe_encoder = WOEEncoder(cols=cat_columns)
    woe_encoder.fit(train_df[cat_columns], train_df[target_column])

    # エンコーディングを適用
    train_df_woe = woe_encoder.transform(train_df[cat_columns])
    test_df_woe = woe_encoder.transform(test_df[cat_columns])

    # 元のデータフレームに新しい特徴量を追加
    for col in cat_columns:
        new_col_name = f'{col}_woe'
        train_df[new_col_name] = train_df_woe[col]
        test_df[new_col_name] = test_df_woe[col]

    # 元のカテゴリ変数を削除
    train_df = train_df.drop(columns=cat_columns)
    test_df = test_df.drop(columns=cat_columns)

    # 結果を結合
    df_encoded = pd.concat([train_df, test_df], axis=0)

    print_with_timestamp('カテゴリ変数のCatBoost, WOEエンコーディング完了')
    return df_encoded, ce_encoder, woe_encoder

def add_nonlinear_features(df):
    print_with_timestamp('数値変数の非線形変換を開始します...')

    # 数値列を選択（SK_ID_CURR と TARGET を除く）
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['SK_ID_CURR', 'TARGET']]
    
    for col in numeric_cols:
        # オリジナルの値の範囲を保存
        original_min = df[col].min()
        original_max = df[col].max()
        
        # スケーリング（0から1の範囲に）
        df[f'{col}_SCALED'] = (df[col] - original_min) / (original_max - original_min)
        
        # 2乗
        df[f'{col}_SQUARED'] = df[f'{col}_SCALED'] ** 2
        
        # 3乗
        df[f'{col}_CUBE'] = df[f'{col}_SCALED'] ** 3
        
        # 平方根
        df[f'{col}_SQRT'] = np.sqrt(df[f'{col}_SCALED'])
        
        # 対数変換 (1を加えて負の値を避ける)
        df[f'{col}_LOG'] = np.log1p(df[f'{col}_SCALED'])
        
        # 指数変換
        df[f'{col}_EXP'] = np.exp(df[f'{col}_SCALED']) - 1
        
        # 元のスケールに戻す
        for suffix in ['SQUARED', 'CUBE', 'SQRT', 'LOG', 'EXP']:
            df[f'{col}_{suffix}'] = df[f'{col}_{suffix}'] * (original_max - original_min) + original_min
        
        # スケーリングした列を削除
        df = df.drop(columns=[f'{col}_SCALED'])
    
    print_with_timestamp('数値変数の非線形変換が完了しました')
    return df

# def add_features_parallel(df, n_neighbors_list=range(100, 1001, 100)):
def add_features_parallel(df, n_neighbors_list=range(500, 501, 100)):
    print_with_timestamp('NearestNeighbors特徴量の計算を開始します...')

    features = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'credit_annuity_ratio']
    df_neighbors = df[features].fillna(0)
    
    def calculate_neighbor_mean(n_neighbors):
        print_with_timestamp(f'{n_neighbors}近傍の計算を開始します...')
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(df_neighbors)
        distances, indices = nbrs.kneighbors(df_neighbors)
        
        def get_neighbor_mean(i):
            return df.loc[df.index[indices[i]]]['TARGET'].mean()
        
        neighbor_means = Parallel(n_jobs=-1)(delayed(get_neighbor_mean)(i) for i in range(len(indices)))
        result = pd.Series(neighbor_means, name=f'neighbors_target_mean_{n_neighbors}', index=df.index)
        
        # 結果をCSVファイルに保存
        result_df = pd.DataFrame({'SK_ID_CURR': df['SK_ID_CURR'], f'neighbors_target_mean_{n_neighbors}': result})
        result_df.to_csv(f'C:/Gdrive/data2/neighbors_target_mean_{n_neighbors}.csv', index=False)
        
        print_with_timestamp(f'{n_neighbors}近傍の計算が完了し、CSVファイルに保存しました')
        return result
    
    new_features = []
    for n in n_neighbors_list:
        new_feature = calculate_neighbor_mean(n)
        new_features.append(new_feature)
    
    result_df = pd.concat([df] + new_features, axis=1)

    print_with_timestamp('NearestNeighbors特徴量の計算が完了しました')
    return result_df

def add_neighbors_target_mean_multiple(df):
    print_with_timestamp('NearestNeighbors特徴量の追加を開始します...')

    # 既存のneighbors_target_mean_*列を削除
    columns_to_drop = [col for col in df.columns if col.startswith('neighbors_target_mean_')]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, axis=1)
        print_with_timestamp(f'{len(columns_to_drop)}個の既存のneighbors_target_mean_*列を削除しました')

    # 重複列の確認と処理
    duplicate_columns = df.columns[df.columns.duplicated()]
    if not duplicate_columns.empty:
        print(f"Warning: 重複列が見つかりました: {duplicate_columns.tolist()}")
        # 重複列の処理（例：最初の列以外を削除）
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        print("重複列を削除しました。")

    # SK_ID_CURR列の一意性を確認
    if df['SK_ID_CURR'].duplicated().any():
        print("Warning: SK_ID_CURR列に重複があります。最初の出現を保持します。")
        df = df.drop_duplicates(subset='SK_ID_CURR', keep='first')

    # CSVファイルを検索
    csv_files = [f for f in os.listdir('C:/Gdrive/data2/') if f.startswith('neighbors_target_mean_') and f.endswith('.csv')]
    
    for csv_file in sorted(csv_files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        print_with_timestamp(f'{csv_file}の読み込みを開始します...')
        neighbors_target_mean_df = pd.read_csv(f'C:/Gdrive/data2/{csv_file}')
        
        # データフレームにneighbors_target_mean_*を追加
        column_name = f"neighbors_target_mean_{csv_file.split('_')[-1].split('.')[0]}"
        df = pd.merge(df, neighbors_target_mean_df[['SK_ID_CURR', column_name]], on='SK_ID_CURR', how='left')
        print_with_timestamp(f'{csv_file}の読み込みと追加が完了しました')
    
    print_with_timestamp('NearestNeighbors特徴量の追加が完了しました')
    return df

def remove_redundant_features(df, correlation_threshold=0.95, variance_threshold=0.01):
    print_with_timestamp('冗長な特徴量の削除を実行します...')

    # 'TARGET'と'SK_ID_CURR'列を一時的に除外
    target_column = df['TARGET'] if 'TARGET' in df.columns else None
    id_column = df['SK_ID_CURR']
    df_features = df.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')
    
    # Step 1: 完全に同一の特徴量を削除
    df_unique = df_features.T.drop_duplicates().T
    print_with_timestamp(f'{df_features.shape[1] - df_unique.shape[1]}個の完全に重複した特徴量を削除しました')
    
    # # Step 2: 高い相関を持つ特徴量の削除
    # correlation_matrix = df_unique.corr().abs()
    # upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    # to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    # df_uncorrelated = df_unique.drop(to_drop, axis=1)
    # print_with_timestamp(f'{len(to_drop)}個の高相関な特徴量を削除しました')
    
    # # Step 3: 低分散の特徴量の削除
    # selector = VarianceThreshold(threshold=variance_threshold)
    # selector.fit(df_uncorrelated)
    # df_high_variance = df_uncorrelated.iloc[:, selector.get_support(indices=True)]
    # print_with_timestamp(f'{df_uncorrelated.shape[1] - df_high_variance.shape[1]}個の低分散な特徴量を削除しました')
    
    print_with_timestamp(f'元の特徴量数: {df_features.shape[1]}, 削除後の特徴量数: {df_unique.shape[1]}')
    
    # 'TARGET'と'SK_ID_CURR'列を再結合
    result = pd.concat([id_column, df_unique], axis=1)
    if target_column is not None:
        result = pd.concat([result, target_column], axis=1)
    
    print_with_timestamp('冗長な特徴量の削除が完了しました')
    return result

def check_and_clean_data(df):
    print_with_timestamp('データのチェックとクリーニングを開始します...')
    
    # 無限大の値をNaNに置換
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 極端に大きな値（例：float64の最大値の99%以上）を持つ列を特定
    extreme_columns = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].abs().max() > 0.99 * np.finfo(np.float64).max:
            extreme_columns.append(col)
            print(f"極端に大きな値を含む列: {col}")
    
    # 極端に大きな値を持つ列を処理（例：99.9パーセンタイルでクリッピング）
    for col in extreme_columns:
        upper_bound = df[col].quantile(0.999)
        lower_bound = df[col].quantile(0.001)
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    print_with_timestamp('データのチェックとクリーニングが完了しました')
    return df

def select_important_features(df, target_col='TARGET', n_features=20):
    print_with_timestamp('重要な特徴量の選択を開始します...')
    
    # データのチェックとクリーニング
    df = check_and_clean_data(df)
    
    train_df = df[df[target_col].notnull()].copy()
    X = train_df.drop(columns=[target_col, 'SK_ID_CURR'])
    y = train_df[target_col]
    
    # 欠損値を一時的に-999で埋める
    X_temp = X.fillna(-999)
    
    mi_scores = mutual_info_classif(X_temp, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    top_features = mi_scores.nlargest(n_features).index.tolist()
    
    print(f"選択された上位{n_features}個の特徴量: {top_features}")
    print_with_timestamp('重要な特徴量の選択が完了しました')
    return top_features

class MemoryEfficientRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        for i in range(self.n_estimators):
            tree = RandomForestClassifier(n_estimators=1, random_state=self.random_state+i if self.random_state else None)
            tree.fit(X, y)
            self.estimators_.append(tree)
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.round(np.mean(predictions, axis=0)).astype(int)

    def predict_proba(self, X):
        probas = np.array([tree.predict_proba(X) for tree in self.estimators_])
        return np.mean(probas, axis=0)

def advanced_imputation(df, columns_to_impute):
    print_with_timestamp('高度な欠損値補完を開始します...')
    
    # 数値型とカテゴリ型の列を分ける
    num_columns = df[columns_to_impute].select_dtypes(include=['int64', 'float64']).columns
    cat_columns = df[columns_to_impute].select_dtypes(include=['object']).columns
    
    # 数値型の列の処理
    if len(num_columns) > 0:
        num_df = df[num_columns]
        it_imputer = IterativeImputer(estimator=MemoryEfficientRandomForestClassifier(n_estimators=10), 
                                      max_iter=10, random_state=0)
        df[num_columns] = it_imputer.fit_transform(num_df)
    
    # カテゴリ型の列の処理
    if len(cat_columns) > 0:
        cat_df = df[cat_columns].copy()
        
        # カテゴリ変数を数値にエンコード
        for col in cat_columns:
            cat_df[col] = pd.Categorical(cat_df[col]).codes
        
        it_imputer = IterativeImputer(estimator=MemoryEfficientRandomForestClassifier(n_estimators=10), 
                                      max_iter=10, random_state=0)
        cat_df_imputed = it_imputer.fit_transform(cat_df)
        
        # 数値を元のカテゴリに戻す
        for i, col in enumerate(cat_columns):
            df[col] = pd.Categorical.from_codes(cat_df_imputed[:, i].astype(int), 
                                                categories=pd.Categorical(df[col]).categories)
    
    print_with_timestamp('高度な欠損値補完が完了しました')
    return df

def add_interaction_features(df, n=20, target_col='TARGET', chunk_size=100000):
    print_with_timestamp('特徴量の交互作用の追加を開始します...')
    
    # 重要な特徴量を選択
    top_features = select_important_features(df, target_col, n)
    
    # データを分割して処理
    chunks = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]
    result_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"チャンク {i+1}/{len(chunks)} を処理中...")
        
        # 選択された特徴量の欠損値を補完
        chunk_imputed = advanced_imputation(chunk[top_features].copy(), top_features)
        
        # 交互作用特徴量の作成
        for col1, col2 in combinations(top_features, 2):
            if chunk_imputed[col1].dtype in ['int64', 'float64'] and chunk_imputed[col2].dtype in ['int64', 'float64']:
                new_col_name = f'{col1}_{col2}_INTERACTION'
                chunk[new_col_name] = chunk_imputed[col1] * chunk_imputed[col2]
                print(f"作成された交互作用特徴量: {new_col_name}")
        
        result_chunks.append(chunk)
    
    # 結果を結合
    df = pd.concat(result_chunks, axis=0)
    
    print_with_timestamp('特徴量の交互作用の追加が完了しました')
    print(f"交互作用追加後のデータフレームの形状: {df.shape}")
    return df

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

def fit_lgbm_gpu(X, y, cv, params: dict=None, verbose=100):
    print_with_timestamp('LightGBMで学習を行います...')

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

    print_with_timestamp('LightGBMの学習完了')
    return oof_preds, models

def evaluate_combinations(df, target_col='TARGET'):
    neighbors_cols = [col for col in df.columns if col.startswith('neighbors_target_mean_')]
    results = []

    # TARGETとSK_ID_CURRを除外した特徴量を取得
    feature_cols = [col for col in df.columns if col not in [target_col, 'SK_ID_CURR']]

    for r in range(1, len(neighbors_cols) + 1):
        for cols in tqdm(combinations(neighbors_cols, r), desc=f'Evaluating {r} columns'):
            # neighbors_target_mean_* 列と他の特徴量を組み合わせる
            selected_cols = list(cols) + [col for col in feature_cols if col not in cols]
            X = df[selected_cols]
            y = df[target_col]

            fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
            cv = list(fold.split(X, y))

            oof, _ = fit_lgbm_gpu(X.values, y.values, cv=cv, params=lgbm_best_param)
            score = roc_auc_score(y, oof)

            results.append({
                'columns': cols,
                'num_columns': len(cols),
                'score': score
            })

    return sorted(results, key=lambda x: x['score'], reverse=True)

def print_top_results(results, top_n=10):
    print(f"\nTop {top_n} combinations:")
    for i, result in enumerate(results[:top_n], 1):
        print(f"{i}. Score: {result['score']:.6f}, Num columns: {result['num_columns']}")
        print(f"   Columns: {', '.join(result['columns'])}")
        print()


def main():
    # ログファイルを初期化
    with open("C:/Gdrive/data2/results.log", "w") as f:
        pass

    # 標準出力をログファイルにリダイレクト（追記）しながら、ターミナルにも出力する
    sys.stdout = DualLogger("C:/Gdrive/data2/results.log", sys.stdout)

    if SKIP_PREPROCESSING:
        train, test = load_preprocessed_data()
    else:
        # 既存の前処理コード
        print_with_timestamp('データを読み込みます...')
        train_df = pd.read_csv('C:/Gdrive/data2/train.csv')
        test_df = pd.read_csv('C:/Gdrive/data2/test.csv')
        sample_submission = pd.read_csv('C:/Gdrive/data2/sample_submission.csv')
        df_all = pd.concat([train_df, test_df], axis=0)
        print_with_timestamp('データ読み込み完了')
        print("データフレームの形状:", df_all.shape)

        # 不要な特徴量を削除します
        df_all = to_drop(df_all)
        print("データフレームの形状:", df_all.shape)

        # 主要な特徴量を追加します
        df_all = to_add_feature(df_all)
        print("データフレームの形状:", df_all.shape)

        # 同一ユーザー特徴量を追加します
        df_all = add_leak_features(df_all)
        print("データフレームの形状:", df_all.shape)

        # 元のカテゴリカル変数を保持した2種のエンコーディングを適用
        df_all = apply_all_encodings(df_all, keep_original=True)
        print("データフレームの形状:", df_all.shape)

        # カテゴリ変数のtarget encodingを適用
        cat_columns = [col for col in df_all.columns if df_all[col].dtype == 'object']
        df_all = improved_target_encode_features(df_all, cat_columns)
        print("データフレームの形状:", df_all.shape)

        # 追加のカテゴリ変数エンコーディング（CatBoost, WOE）を開始します
        df_all, ce_encoder, woe_encoder = improved_categorical_encoding(df_all)
        print("データフレームの形状:", df_all.shape)

        # # 数値変数の非線形変換を開始します
        # df_all = add_nonlinear_features(df_all)
        # print("データフレームの形状:", df_all.shape)

        # 冗長な特徴量の削除を実行します
        df_all = remove_redundant_features(df_all)
        print("データフレームの形状:", df_all.shape)

        # # 特徴量の交互作用を追加します
        # df_all = add_interaction_features(df_all, top_features=[], n=10)
        # print("データフレームの形状:", df_all.shape)

        # # NearestNeighbors特徴量の計算を開始します
        # df_all = add_features_parallel(df_all)
        # print("データフレームの形状:", df_all.shape)

        # NearestNeighbors特徴量をcsvから追加します
        df_all = add_neighbors_target_mean_multiple(df_all)
        print("データフレームの形状:", df_all.shape)

        # データをtrainとtestに分割
        train = df_all[df_all.loc[:, 'SK_ID_CURR'] < 171202]
        test = df_all[df_all.loc[:, 'SK_ID_CURR'] > 171201]

        # データをCSVファイルに保存
        print_with_timestamp('前処理済みデータをCSVファイルに保存します...')
        train.to_csv('C:/Gdrive/data2/train_processed.csv', index=False)
        test.to_csv('C:/Gdrive/data2/test_processed.csv', index=False)
        print_with_timestamp('保存完了')

    if OPTIMIZE_NEIGHBORS_COLUMNS:
        # neighbors_target_mean列の組み合わせを評価
        print_with_timestamp('neighbors_target_mean列の組み合わせを評価します...')
        train_for_eval = train.drop(columns=['SK_ID_CURR'])  # SK_ID_CURRを除外
        results = evaluate_combinations(train_for_eval)
        print_top_results(results)

        # 最良の組み合わせを使用してモデルを学習・予測
        best_combination = results[0]['columns']
        print_with_timestamp(f'最良の組み合わせを使用してモデルを学習します: {best_combination}')

        # 最良の組み合わせと他の特徴量を組み合わせる
        selected_cols = list(best_combination) + [col for col in train.columns if col not in best_combination and col not in ['TARGET', 'SK_ID_CURR']]
    else:
        # 全ての列を使用（TARGETとSK_ID_CURRを除く）
        selected_cols = [col for col in train.columns if col not in ['TARGET', 'SK_ID_CURR']]
        print_with_timestamp('全ての特徴量を使用してモデルを学習します')
        
    train_x = train[selected_cols]
    train_y = train['TARGET']
    test_x = test[selected_cols]
    print("train_xの形状:", train_x.shape)
    print("test_xの形状:", test_x.shape)

    X = train_x.values
    y = train_y.values

    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
    cv = list(fold.split(X, y))

    # メインの予測部分
    oof, models = fit_lgbm_gpu(X, y, cv=cv, params=lgbm_best_param)

    pred = np.array([model.predict(test_x.values) for model in models])  # Changed from predict_proba to predict
    pred = np.mean(pred, axis=0)

    submission = sample_submission.copy()
    submission['TARGET'] = pred

    print_with_timestamp('予測結果を出力します...')
    submission.to_csv('C:/Gdrive/data2/submission.csv', index=False)
    print_with_timestamp('予測結果の出力完了')


if __name__ == "__main__":
    main()