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

from sklearn.model_selection import train_test_split
import optuna

import os
import datetime
import time
last_time = None  # 前回の時間を保持するグローバル変数

# グローバル変数の設定
SKIP_PREPROCESSING = False  # Trueに変更すると前処理をスキップします
DATA_DIR = 'C:/Gdrive/data2/'
global X, y

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

def load_preprocessed_data():
    print_with_timestamp('前処理済みデータを読み込みます...')
    train = pd.read_csv(os.path.join(DATA_DIR, 'train_processed_nn.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test_processed_nn.csv'))
    print_with_timestamp('前処理済みデータの読み込み完了')
    return train, test

def preprocess_data():
    print_with_timestamp("データの読み込みと前処理を開始します...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train_processed.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test_processed.csv'))
    df_all = pd.concat([train, test], axis=0, sort=False)
    print("Initial df_all shape:", df_all.shape)

    # user_target_mean カラムを削除（欠損値多数で補完困難）
    if 'user_target_mean' in df_all.columns:
        df_all = df_all.drop('user_target_mean', axis=1)
        print("Dropped user_target_mean column")

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

    # データの保存
    print_with_timestamp("Saving processed data...")
    train.to_csv(os.path.join(DATA_DIR, 'train_processed_nn.csv'), index=False)
    test.to_csv(os.path.join(DATA_DIR, 'test_processed_nn.csv'), index=False)

    return train, test

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

def improve_missing_values(df, batch_size=500):
    columns_to_exclude = ['SK_ID_CURR', 'TARGET']
    id_column = df['SK_ID_CURR'].copy()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.drop(columns_to_exclude, errors='ignore')
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.drop(columns_to_exclude, errors='ignore')
    
    all_columns = list(numeric_columns) + list(categorical_columns)
    is_numeric = [True] * len(numeric_columns) + [False] * len(categorical_columns)
    
    num_cores = max(1, psutil.cpu_count(logical=False) - 1)  # 物理コア数を使用
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    results = {col: [] for col in all_columns}
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        for i in range(0, len(df), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{len(df)//batch_size + 1}")
            batch = df.iloc[i:i+batch_size]
            
            futures = [executor.submit(impute_column, (col, batch[col].values, is_num)) for col, is_num in zip(all_columns, is_numeric)]
            
            for future in as_completed(futures):
                col, imputed_data = future.result()
                results[col].extend(imputed_data)
            
            # メモリ使用量をチェック
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > 80:
                print(f"Warning: High memory usage ({memory_usage}%). Consider reducing batch_size.")
    
    # 結果をデータフレームに適用
    for col in all_columns:
        if len(results[col]) == len(df):
            df[col] = results[col]
        else:
            print(f"Warning: Column {col} has {len(results[col])} values, expected {len(df)}. Skipping this column.")
    
    df['SK_ID_CURR'] = id_column
    df = df.set_index('SK_ID_CURR').sort_index().reset_index()
    
    return df

def convert_bool_to_int(df):
    bool_columns = df.select_dtypes(include=['bool']).columns
    for col in bool_columns:
        df[col] = df[col].astype(int)
    return df

def objective(trial):
    global X, y  # グローバル変数を使用することを宣言

    # ハイパーパラメータの探索範囲を定義
    params = {
        'n_d': trial.suggest_int('n_d', 9, 11),
        'n_a': trial.suggest_int('n_a', 33, 35),
        'n_steps': trial.suggest_int('n_steps', 3, 5),
        'gamma': trial.suggest_float('gamma', 1.28, 1.32),
        'n_independent': trial.suggest_int('n_independent', 3, 5),
        'n_shared': trial.suggest_int('n_shared', 2, 4),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1.5e-5, log=True),
        'momentum': trial.suggest_float('momentum', 0.18, 0.19),
        'clip_value': trial.suggest_float('clip_value', 0.01, 0.015, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.009, 0.012, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
        'virtual_batch_size': trial.suggest_categorical('virtual_batch_size', [32, 64, 128])
    }

    # クロスバリデーションの設定
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = TabNetClassifier(
            n_d=params['n_d'],
            n_a=params['n_a'],
            n_steps=params['n_steps'],
            gamma=params['gamma'],
            n_independent=params['n_independent'],
            n_shared=params['n_shared'],
            lambda_sparse=params['lambda_sparse'],
            momentum=params['momentum'],
            clip_value=params['clip_value'],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=params['learning_rate']),
            scheduler_params={"step_size":25, "gamma":0.97},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax'
        )

        model.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            max_epochs=100,
            patience=10,
            batch_size=params['batch_size'],
            virtual_batch_size=params['virtual_batch_size'],
            num_workers=0,
            drop_last=False
        )

        y_pred = model.predict_proba(X_val.values)[:, 1]
        fold_score = roc_auc_score(y_val, y_pred)
        scores.append(fold_score)

        trial.report(fold_score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores)

def main():
    global X, y  # グローバル変数を使用することを宣言

    # ログファイルを初期化
    with open("C:/Gdrive/data2/results_tabnet.log", "w") as f:
        pass
    # 標準出力をログファイルにリダイレクト（追記）しながら、ターミナルにも出力する
    sys.stdout = DualLogger("C:/Gdrive/data2/results_tabnet.log", sys.stdout)

    if SKIP_PREPROCESSING:
        train, test = load_preprocessed_data()
    else:
        train, test = preprocess_data()

    # 特徴量とターゲットの分離
    print_with_timestamp("Separating features and target...")
    X = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y = train['TARGET']
    X_test = test.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    print("Final X shape:", X.shape)
    print("Final y shape:", y.shape)
    print("Final X_test shape:", X_test.shape)
    print(X.dtypes.value_counts())

    # Optunaの設定
    study = optuna.create_study(direction='maximize', 
                                pruner=optuna.pruners.MedianPruner(),
                                sampler=optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24))
    study.optimize(objective, n_trials=300, timeout=3600*24)  # 24時間の制限

    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value:', trial.value)
    print('  Params:')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # 最適なハイパーパラメータでモデルを学習
    best_params = trial.params
    final_model = TabNetClassifier(
        n_d=best_params['n_d'],
        n_a=best_params['n_a'],
        n_steps=best_params['n_steps'],
        gamma=best_params['gamma'],
        n_independent=best_params['n_independent'],
        n_shared=best_params['n_shared'],
        lambda_sparse=best_params['lambda_sparse'],
        momentum=best_params['momentum'],
        clip_value=best_params['clip_value'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=best_params['learning_rate']),
        scheduler_params={"step_size":25, "gamma":0.97},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax'
    )
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    final_model.fit(
        X_train=X_train.values, y_train=y_train.values,
        eval_set=[(X_train.values, y_train.values), (X_val.values, y_val.values)],
        eval_name=['train', 'valid'],
        max_epochs=150,
        patience=15,
        batch_size=best_params['batch_size'],
        virtual_batch_size=best_params['virtual_batch_size'],
        num_workers=0,
        drop_last=False
    )

    # テストデータの予測
    test_predictions = final_model.predict_proba(X_test.values)[:, 1]

    # 提出用のデータフレーム作成
    submission = pd.DataFrame({'SK_ID_CURR': test['SK_ID_CURR'], 'TARGET': test_predictions})
    submission.to_csv('C:/Gdrive/data2/submission_tabnet.csv', index=False)
    print("Submission file created: submission_tabnet.csv")

if __name__ == '__main__':
    main()