import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr

from sklearn.model_selection import StratifiedKFold

def select_features(X, y, n_features=200, correlation_threshold=0.95):
    # 特徴量の重要度を計算
    mi_scores = mutual_info_classif(X, y)
    mi_scores = pd.Series(mi_scores, index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    # 最も重要な特徴量を選択
    selected_features = mi_scores.head(n_features).index.tolist()

    # 相関の高い特徴量を除去
    X_selected = X[selected_features]
    corr_matrix = X_selected.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    
    final_features = [col for col in selected_features if col not in to_drop]

    print(f"Selected {len(final_features)} features")
    return final_features

# メインの処理
# データの読み込み
train = pd.read_csv('C:/Gdrive/data2/train_processed.csv')
test = pd.read_csv('C:/Gdrive/data2/test_processed.csv')

# 特徴量とターゲットの分離
X = train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = train['TARGET']
X_test = test.drop(['SK_ID_CURR'], axis=1)

# 無限大の値をNaNに置き換える
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# 完全に欠損している列を削除
X = X.dropna(axis=1, how='all')
X_test = X_test[X.columns]

# 欠損値の補完
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 極端に大きな値や小さな値をクリッピング
for column in X.columns:
    lower_bound = X[column].quantile(0.001)
    upper_bound = X[column].quantile(0.999)
    X[column] = X[column].clip(lower_bound, upper_bound)
    X_test[column] = X_test[column].clip(lower_bound, upper_bound)

# 特徴量選択
selected_features = select_features(X, y, n_features=200, correlation_threshold=0.95)

# 選択された特徴量のみを使用
X = X[selected_features]
X_test = X_test[selected_features]

# クロスバリデーションの設定
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=69)

# 結果を格納するリスト
oof_predictions = np.zeros(len(X))
test_predictions = np.zeros(len(X_test))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # データの標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
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
        X_train=X_train_scaled, y_train=y_train,
        eval_set=[(X_val_scaled, y_val)],
        max_epochs=100, patience=10,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # バリデーションデータの予測
    oof_predictions[val_idx] = model.predict_proba(X_val_scaled)[:, 1]
    
    # テストデータの予測
    X_test_scaled = scaler.transform(X_test)
    test_predictions += model.predict_proba(X_test_scaled)[:, 1] / n_splits
    
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
submission.to_csv('C:/Gdrive/data2/tabnet_submission.csv', index=False)
print("Submission file created: tabnet_submission.csv")

# 削除された列の数を出力
print(f"Number of columns dropped: {len(train.columns) - len(X.columns) - 2}")  # -2 for 'TARGET' and 'SK_ID_CURR'