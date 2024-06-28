import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy import stats
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count
import psutil

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

def remove_all_nan_columns(df):
    return df.dropna(axis=1, how='all')

def optimize_outliers(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(['SK_ID_CURR', 'TARGET'], errors='ignore')
    
    # 無限大の値をNaNに置き換える
    df = df.replace([np.inf, -np.inf], np.nan)
    
    for col in numeric_columns:
        # NaNを無視してZ-scoreを計算
        z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
        df[col] = df[col].mask(z_scores > 3, df[col].median())
        
        # IQRを使用して外れ値を特定
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # StandardScalerを使用してスケーリング
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df

def impute_column(args):
    col, data, is_numeric = args
    if is_numeric:
        imputer = KNNImputer(n_neighbors=5)
        imputed_data = imputer.fit_transform(data.reshape(-1, 1)).ravel()
    else:
        # カテゴリカル変数の場合は最頻値で補完
        imputed_data = pd.Series(data).fillna(pd.Series(data).mode()[0]).values
    return col, imputed_data

def improve_missing_values(df, batch_size=1000):
    columns_to_exclude = ['SK_ID_CURR', 'TARGET']
    id_column = df['SK_ID_CURR'].copy()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(columns_to_exclude, errors='ignore')
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.drop(columns_to_exclude, errors='ignore')
    
    all_columns = list(numeric_columns) + list(categorical_columns)
    is_numeric = [True] * len(numeric_columns) + [False] * len(categorical_columns)
    
    num_cores = max(1, cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    results = {}
    with Pool(num_cores) as pool:
        for i in range(0, len(df), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{len(df)//batch_size + 1}")
            batch = df.iloc[i:i+batch_size]
            
            batch_results = pool.map(impute_column, [(col, batch[col].values, is_num) for col, is_num in zip(all_columns, is_numeric)])
            
            for col, imputed_data in batch_results:
                if col not in results:
                    results[col] = []
                results[col].extend(imputed_data)
            
            # メモリ使用量をチェック
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 80:
                print(f"Warning: High memory usage ({memory_usage}%). Consider reducing batch_size.")
    
    for col in all_columns:
        df[col] = results[col]
    
    df['SK_ID_CURR'] = id_column
    df = df.set_index('SK_ID_CURR').sort_index().reset_index()
    
    return df

def convert_bool_to_int(df):
    bool_columns = df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
    return df

if __name__ == '__main__':
    # ログファイルを初期化
    with open("C:/Gdrive/data2/results_tabnet.log", "w") as f:
        pass

    # 標準出力をログファイルにリダイレクト（追記）しながら、ターミナルにも出力する
    sys.stdout = DualLogger("C:/Gdrive/data2/results_tabnet.log", sys.stdout)

    # データの読み込みと前処理
    print_with_timestamp("Loading data...")
    train = pd.read_csv('C:/Gdrive/data2/train_processed.csv')
    test = pd.read_csv('C:/Gdrive/data2/test_processed.csv')
    df_all = pd.concat([train, test], axis=0, sort=False)
    print("Initial df_all shape:", df_all.shape)

    # 完全に欠損している列を削除
    df_all = remove_all_nan_columns(df_all)
    print("After removing all-NaN columns - df_all shape:", df_all.shape)

    # 外れ値の処理
    print_with_timestamp("Optimizing outliers...")
    df_all = optimize_outliers(df_all)
    print("After outlier treatment - df_all shape:", df_all.shape)

    # 欠損値の補完
    print_with_timestamp("Improving missing values...")
    df_all = improve_missing_values(df_all)
    print("After missing value imputation - df_all shape:", df_all.shape)

    # bool型をint型に変換
    print_with_timestamp("Converting bool to int...")
    df_all = convert_bool_to_int(df_all)
    print("After converting bool to int - df_all shape:", df_all.shape)

    # trainとtestに分割
    train = df_all[df_all['TARGET'].notna()].copy()
    test = df_all[df_all['TARGET'].isna()].copy()

    # データの保存★
    print_with_timestamp("Saving processed data...")
    train.to_csv('C:/Gdrive/data2/train_processed_nn.csv', index=False)
    test.to_csv('C:/Gdrive/data2/test_processed_nn.csv', index=False)

    # 特徴量とターゲットの分離
    print_with_timestamp("Separating features and target...")
    X = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = train['TARGET']
    X_test = test.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    print("Final X shape:", X.shape)
    print("Final y shape:", y.shape)
    print("Final X_test shape:", X_test.shape)
    print(X.dtypes.value_counts())

    # クロスバリデーションの設定
    print_with_timestamp("Setting up cross-validation...")
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)

    # 結果を格納するリスト
    oof_predictions = np.zeros(len(X))
    test_predictions = np.zeros(len(X_test))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"Fold {fold}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # TabNetモデルの定義
        model = TabNetClassifier(
            n_d=64, n_a=64, n_steps=5,
            gamma=1.5, n_independent=2, n_shared=2,
            lambda_sparse=1e-3, momentum=0.3, clip_value=2.,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax'
        )
        
        # モデルのトレーニング
        model.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            max_epochs=100, patience=10,
            batch_size=1024, virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        # バリデーションデータの予測
        oof_predictions[val_idx] = model.predict_proba(X_val.values)[:, 1]
        
        # テストデータの予測
        test_predictions += model.predict_proba(X_test.values)[:, 1] / n_splits
        
        # フォールドのスコアを計算
        fold_score = roc_auc_score(y_val, oof_predictions[val_idx])
        fold_scores.append(fold_score)
        print(f"Fold {fold} AUC: {fold_score}")

    # 全体のスコアを計算
    overall_score = roc_auc_score(y, oof_predictions)
    print(f"Overall AUC: {overall_score}")
    print(f"Average Fold AUC: {np.mean(fold_scores)}")

    # 提出用のデータフレーム作成
    submission = pd.DataFrame({'SK_ID_CURR': test['SK_ID_CURR'], 'TARGET': test_predictions})
    submission.to_csv('C:/Gdrive/data2/submission_tabnet.csv', index=False)
    print("Submission file created: submission_tabnet.csv")

    # 削除された列の数を出力
    print(f"Number of columns dropped: {len(train.columns) - len(X.columns) - 2}")  # -2 for 'TARGET' and 'SK_ID_CURR'

