import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

### Reading Data
path = "C:/Gdrive/data2/"

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')

train["train"] = 1
test["train"] = 0
# 邪魔なデータを落とす。
train.drop(train.index[[4042, 39044, 47453, 72207, 74592, 153489]], inplace=True)
combine = pd.concat([train, test])

### 似ている特徴量のグループでの欠損値の個数
days_features = ['DAYS_BIRTH', 'DAYS_EMPLOYED','DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']
amt_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
socialcircle_feature = ["DEF_30_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"]
bureau_feature = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']
ext_feature = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
combine["missing_allcounts"] = combine.isnull().sum(axis=1)
combine["missing_dayscounts"] = combine[days_features].isnull().sum(axis=1)
combine["missing_amtcounts"] = combine[amt_features].isnull().sum(axis=1)
combine["missing_socialcirclecounts"] = combine[socialcircle_feature].isnull().sum(axis=1)
combine["missing_bureaucounts"] = combine[bureau_feature].isnull().sum(axis=1)
combine["missing_extcounts"] = combine[ext_feature].isnull().sum(axis=1)
### 扱いやすいようにDAYS系からYEAR系の特徴量を作っておく。
combine[days_features] = -combine[days_features]

### 欠損値を調べる関数
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

### Pre-Processing

# 数値型とカテゴリ型変数をまとめる
numerical_feats_c = combine.dtypes[combine.dtypes != "object"].index
categorical_feats_c = combine.dtypes[combine.dtypes == "object"].index

# 異常値を欠損値に変える
combine.loc[(combine['DAYS_EMPLOYED'] == -365243)&(combine['NAME_INCOME_TYPE']=="Unemployed"), 'DAYS_EMPLOYED'] = 0
combine.loc[(combine['DAYS_EMPLOYED'] == -365243)&(combine['NAME_INCOME_TYPE']=="Pensioner"), 'DAYS_EMPLOYED'] = np.nan
combine["PENSIONER"] = 0


# 欠損値を埋める
combine.loc[(combine['FLAG_OWN_CAR'].isnull()) & (combine['OWN_CAR_AGE'].isnull()), 'FLAG_OWN_CAR'] = "N"
combine.loc[(combine['FLAG_OWN_CAR'].isnull())&(combine['OWN_CAR_AGE'].notnull()), 'FLAG_OWN_CAR'] = "Y"
combine.loc[combine['OWN_CAR_AGE'] >= 60, 'OWN_CAR_AGE'] = np.nan
combine.loc[(combine['OCCUPATION_TYPE'].isnull())&(combine['DAYS_EMPLOYED']==0), "OCCUPATION_TYPE"] = "XNA"
combine.loc[(combine['OCCUPATION_TYPE'].isnull())&(combine['DAYS_EMPLOYED'].isnull()), "OCCUPATION_TYPE"] = "XNA"

for column in ["OCCUPATION_TYPE", "NAME_TYPE_SUITE", "FLAG_OWN_REALTY"]:
    combine[column].fillna("missing", inplace=True)
    
combine.loc[combine['DAYS_LAST_PHONE_CHANGE'].isnull(), 'DAYS_LAST_PHONE_CHANGE'] = combine['DAYS_LAST_PHONE_CHANGE'].mode()[0]
combine.loc[combine['CNT_FAM_MEMBERS'].isnull(), 'CNT_FAM_MEMBERS'] = combine['CNT_FAM_MEMBERS'].mode()[0]
combine.loc[combine['AMT_ANNUITY'].isnull(), 'AMT_ANNUITY'] = combine['AMT_ANNUITY'].mean()
combine["AMT_REQ_CREDIT_BUREAU_HOUR"] = combine["AMT_REQ_CREDIT_BUREAU_HOUR"].fillna(0)
combine["AMT_REQ_CREDIT_BUREAU_MON"] = combine["AMT_REQ_CREDIT_BUREAU_MON"].fillna(0)
combine["AMT_REQ_CREDIT_BUREAU_QRT"] = combine["AMT_REQ_CREDIT_BUREAU_QRT"].fillna(0)
combine["AMT_REQ_CREDIT_BUREAU_YEAR"] = combine["AMT_REQ_CREDIT_BUREAU_YEAR"].fillna(0)

cols = ["DEF_30_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE","OBS_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE", "DAYS_LAST_PHONE_CHANGE"]
for i in cols :
    combine[i]  = combine[i].fillna(combine[i].mode()[0])

combine["CREDIT_09"] = combine["AMT_CREDIT"] * 0.9
for i in combine[combine["AMT_GOODS_PRICE"].isnull()][["AMT_GOODS_PRICE", "CREDIT_09"]].index: 
    combine["AMT_GOODS_PRICE"][i] = combine["CREDIT_09"][i]
del combine["CREDIT_09"]    

### 四則演算で作る特徴量
# DAYS系
combine["YEAR_EMPLOYED"] = combine["DAYS_EMPLOYED"]/365
combine["YEAR_BIRTH"] = combine["DAYS_BIRTH"]/365
combine["YEAR_REGISTRATION"] = combine["DAYS_REGISTRATION"]/365
combine["YEAR_ID_PUBLISH"] = combine["DAYS_ID_PUBLISH"]/365

combine["YEAR_ID_PUBLISH_2"] = combine["YEAR_ID_PUBLISH"]*combine["YEAR_ID_PUBLISH"]
combine["YEAR_REGISTRATION_2"] = combine["YEAR_REGISTRATION"]*combine["YEAR_REGISTRATION"]
combine["YEAR_BIRTH_2"] = combine["YEAR_BIRTH"]*combine["YEAR_BIRTH"]
combine["YEAR_EMPLOYED_2"] = combine["YEAR_EMPLOYED"]*combine["YEAR_EMPLOYED"]

combine["ID_PUBLISH_REGISTRATION_hiku"] = combine["YEAR_REGISTRATION"] - combine["YEAR_ID_PUBLISH"]

combine["EMPLOYED_BIRTH_mainas"] = combine["DAYS_BIRTH"] - combine["DAYS_EMPLOYED"]
combine["EMPLOYED_BIRTH_waru"] = combine["YEAR_EMPLOYED"]/combine["YEAR_BIRTH"]
combine["REGISTRATION_BIRTH"] = combine["YEAR_REGISTRATION"]/combine["YEAR_BIRTH"]
combine["ID_PUBLISH_BIRTH"] = combine["YEAR_ID_PUBLISH"]/combine["YEAR_BIRTH"]
combine["ID_PUBLISH_REGISTRATION_kake"] = combine["YEAR_REGISTRATION"]*combine["YEAR_ID_PUBLISH"]


# AMT系
combine['INCOME_CREDIT_PERC'] = combine['AMT_INCOME_TOTAL']/combine['AMT_CREDIT']
combine['GOODS_PERC'] = combine['AMT_GOODS_PRICE']/combine['AMT_CREDIT']
combine['RATIO_INCOME_GOODS_hiku'] = combine['AMT_INCOME_TOTAL'] -  combine['AMT_GOODS_PRICE']
combine['ANNUITY_LENGTH_CR'] = combine['AMT_CREDIT']/combine['AMT_ANNUITY']# 借入額と毎月の支払額の比
combine['ANNUITY_LENGTH_CR_2'] = (combine['AMT_CREDIT']/combine['AMT_ANNUITY'])**2# 借入額と毎月の支払額の比

combine["ANNUITY_LENGTH_BIRTH"] = combine["YEAR_BIRTH"] + combine['ANNUITY_LENGTH_CR']
combine['ANNUITY_INCOME_PERCENT'] = combine['AMT_INCOME_TOTAL']/combine['AMT_ANNUITY']

combine['CREDIT_YEAR_ID'] = combine['AMT_CREDIT']/combine["YEAR_ID_PUBLISH"]
combine['CREDIT_YEAR_REGISTRATION'] = combine['AMT_CREDIT']/combine["YEAR_REGISTRATION"]
combine["ANNUITY_LENGTH_ID_PUBLISH"] = combine['ANNUITY_LENGTH_CR']/combine["YEAR_ID_PUBLISH"]


# 金利
combine["kinri"] = combine['AMT_CREDIT'] - combine['AMT_GOODS_PRICE']

# 一人当たりで考える
combine['INCOME_PER_FAMILY_MEMBER'] = combine['AMT_INCOME_TOTAL'] / combine['CNT_FAM_MEMBERS']

# EXT系
combine['app EXT_SOURCE mean'] = combine[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
combine['app EXT_SOURCE std'] = combine[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)
combine["app EXT_SOURCE max"] = combine[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis = 1)
combine["app EXT_SOURCE min"] = combine[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis = 1)
combine['app EXT_SOURCE max_1'] = combine['app EXT_SOURCE max'] -  combine['EXT_SOURCE_1']
combine['app EXT_SOURCE max_2'] = combine['app EXT_SOURCE max'] -  combine['EXT_SOURCE_2']
combine['app EXT_SOURCE max_3'] = combine['app EXT_SOURCE max'] -  combine['EXT_SOURCE_3']
combine['app EXT_SOURCE min_1'] = combine['app EXT_SOURCE min'] -  combine['EXT_SOURCE_1']
combine['app EXT_SOURCE min_2'] = combine['app EXT_SOURCE min'] -  combine['EXT_SOURCE_2']
combine['app EXT_SOURCE min_3'] = combine['app EXT_SOURCE min'] -  combine['EXT_SOURCE_3']

combine['app EXT_SOURCE_1_EXT_SOURCE_2_kake'] = combine['EXT_SOURCE_1'] * combine['EXT_SOURCE_2']
combine['app EXT_SOURCE_1_EXT_SOURCE_3_kake'] = combine['EXT_SOURCE_1'] * combine['EXT_SOURCE_3']
combine['app EXT_SOURCE_2_EXT_SOURCE_3_kake'] = combine['EXT_SOURCE_2'] * combine['EXT_SOURCE_3']

# EXT＋α系

combine['app EXT_SOURCE_1_DAYS_BIRTH_waru'] = combine['EXT_SOURCE_1'] / combine['YEAR_BIRTH']
combine['app EXT_SOURCE_2_DAYS_BIRTH_waru'] = combine['EXT_SOURCE_2'] / combine['YEAR_BIRTH']
combine['app EXT_SOURCE_3_DAYS_BIRTH_waru'] = combine['EXT_SOURCE_3'] / combine['YEAR_BIRTH']

combine.loc[combine["DAYS_EMPLOYED"]==0, "DAYS_EMPLOYED"] = 0.01
combine['app EXT_SOURCE_1_DAYS_EMPLOYED_waru'] = combine['EXT_SOURCE_1'] / combine['YEAR_EMPLOYED']
combine['app EXT_SOURCE_2_DAYS_EMPLOYED_waru'] = combine['EXT_SOURCE_2'] / combine['YEAR_EMPLOYED']
combine['app EXT_SOURCE_3_DAYS_EMPLOYED_waru'] = combine['EXT_SOURCE_3'] / combine['YEAR_EMPLOYED']
combine.loc[combine["DAYS_EMPLOYED"]==0.01, "DAYS_EMPLOYED"] = 0

### ドメイン知識で作る特徴量
# 学歴を就職年齢に変換
combine["YEAR_GRADUATE"] = 0
combine.loc[combine['NAME_EDUCATION_TYPE']=="Lower secondary", 'YEAR_GRADUATE'] = 15
combine.loc[combine['NAME_EDUCATION_TYPE']=="Secondary / secondary special", 'YEAR_GRADUATE'] = 18
combine.loc[combine['NAME_EDUCATION_TYPE']=="Incomplete higher", 'YEAR_GRADUATE'] = 20
combine.loc[combine['NAME_EDUCATION_TYPE']=="Higher education", 'YEAR_GRADUATE'] = 24
combine.loc[combine['NAME_EDUCATION_TYPE']=="Academic degree", 'YEAR_GRADUATE'] = 28


# FICOスコアを再現する
combine["EXT_SOURCE_1_CUT"] = pd.cut(combine["EXT_SOURCE_1"], 550, labels=False)
combine["EXT_SOURCE_2_CUT"] = pd.cut(combine["EXT_SOURCE_2"], 550, labels=False)
combine["EXT_SOURCE_3_CUT"] = pd.cut(combine["EXT_SOURCE_3"], 550, labels=False)
combine.loc[combine['EXT_SOURCE_1_CUT'] <= 259, 'EXT_SOURCE_1_CUT'] = 0
combine.loc[(combine['EXT_SOURCE_1_CUT'] > 260) & (combine['EXT_SOURCE_1_CUT'] <= 359), 'EXT_SOURCE_1_CUT'] = 1
combine.loc[(combine['EXT_SOURCE_1_CUT'] > 360) & (combine['EXT_SOURCE_1_CUT'] <= 424), 'EXT_SOURCE_1_CUT'] = 2
combine.loc[(combine['EXT_SOURCE_1_CUT'] > 425) & (combine['EXT_SOURCE_1_CUT'] <= 459), 'EXT_SOURCE_1_CUT'] = 3
combine.loc[combine['EXT_SOURCE_1_CUT'] > 459, 'EXT_SOURCE_1_CUT'] = 4

combine.loc[combine['EXT_SOURCE_2_CUT'] <= 259, 'EXT_SOURCE_2_CUT'] = 0
combine.loc[(combine['EXT_SOURCE_2_CUT'] > 260) & (combine['EXT_SOURCE_2_CUT'] <= 359), 'EXT_SOURCE_2_CUT'] = 1
combine.loc[(combine['EXT_SOURCE_2_CUT'] > 360) & (combine['EXT_SOURCE_2_CUT'] <= 424), 'EXT_SOURCE_2_CUT'] = 2
combine.loc[(combine['EXT_SOURCE_2_CUT'] > 425) & (combine['EXT_SOURCE_2_CUT'] <= 459), 'EXT_SOURCE_2_CUT'] = 3
combine.loc[combine['EXT_SOURCE_2_CUT'] > 459, 'EXT_SOURCE_2_CUT'] = 4

combine.loc[combine['EXT_SOURCE_3_CUT'] <= 259, 'EXT_SOURCE_3_CUT'] = 0
combine.loc[(combine['EXT_SOURCE_3_CUT'] > 260) & (combine['EXT_SOURCE_3_CUT'] <= 359), 'EXT_SOURCE_3_CUT'] = 1
combine.loc[(combine['EXT_SOURCE_3_CUT'] > 360) & (combine['EXT_SOURCE_3_CUT'] <= 424), 'EXT_SOURCE_3_CUT'] = 2
combine.loc[(combine['EXT_SOURCE_3_CUT'] > 425) & (combine['EXT_SOURCE_3_CUT'] <= 459), 'EXT_SOURCE_3_CUT'] = 3
combine.loc[combine['EXT_SOURCE_3_CUT'] > 459, 'EXT_SOURCE_3_CUT'] = 4
combine['EXT_SOURCE_CUT_MEAN'] = combine[['EXT_SOURCE_1_CUT', 'EXT_SOURCE_2_CUT', 'EXT_SOURCE_3_CUT']].mean(axis = 1)

# socialcircle_feature,bureau_featureの二値化
combine["NO_DEF_30L_CIRCLE"] = 0
combine.loc[combine["DEF_30_CNT_SOCIAL_CIRCLE"]>=1, 'NO_DEF_30L_CIRCLE'] = 1
combine["NO_DEF_60L_CIRCLE"] = 0
combine.loc[combine["DEF_60_CNT_SOCIAL_CIRCLE"]>=1, 'NO_DEF_60L_CIRCLE'] = 1
combine["NO_OBS_30L_CIRCLE"] = 0
combine.loc[combine["OBS_30_CNT_SOCIAL_CIRCLE"]>=1, 'NO_OBS_30L_CIRCLE'] = 1
combine["NO_OBS_60L_CIRCLE"] = 0
combine.loc[combine["OBS_60_CNT_SOCIAL_CIRCLE"]>=1, 'NO_OBS_60L_CIRCLE'] = 1
combine["NO_BUREAU_HOUR"] = 0
combine.loc[combine["AMT_REQ_CREDIT_BUREAU_HOUR"]>=1, 'NO_BUREAU_HOUR'] = 1
combine["NO_BUREAU_MON"] = 0
combine.loc[combine["AMT_REQ_CREDIT_BUREAU_MON"]>=1, 'NO_BUREAU_MON'] = 1
combine["NO_BUREAU_QRT"] = 0
combine.loc[combine["AMT_REQ_CREDIT_BUREAU_QRT"]>=1, 'NO_BUREAU_QRT'] = 1
combine["NO_BUREAU_YEAR"] = 0
combine.loc[combine["AMT_REQ_CREDIT_BUREAU_YEAR"]>=1, 'NO_BUREAU_YEAR'] = 1

### Groupbyで作る特徴量１軍（ORGANIZATIONとOCCUPATION）
# 特徴量の寄与が大きいものから１〜３軍とする。
combine['INCOME_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].transform('min')
combine['INCOME_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL'].transform('min')


combine['CREDIT_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_CREDIT'].transform('mean')
combine['CREDIT_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_CREDIT'].transform('mean')
combine['CREDIT_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_CREDIT'].transform('max')
combine['CREDIT_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_CREDIT'].transform('max')
combine['CREDIT_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_CREDIT'].transform('min')
combine['CREDIT_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_CREDIT'].transform('min')

combine['ANNUITY_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_ANNUITY'].transform('min')
combine['ANNUITY_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_ANNUITY'].transform('min')

combine['GOODS_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['AMT_GOODS_PRICE'].transform('min')
combine['GOODS_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['AMT_GOODS_PRICE'].transform('min')


combine['EXT_SOURCE_1_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_1_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_1_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_1'].transform('min')
combine['EXT_SOURCE_1_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_1'].transform('min')

combine['EXT_SOURCE_2_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_2_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_2_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_2'].transform('min')
combine['EXT_SOURCE_2_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_2'].transform('min')


combine['EXT_SOURCE_3_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_3'].transform('mean')
combine['EXT_SOURCE_3_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_3'].transform('mean')
combine['EXT_SOURCE_3_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EXT_SOURCE_3'].transform('min')
combine['EXT_SOURCE_3_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EXT_SOURCE_3'].transform('min')


combine['ANNUITY_LEN_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['ANNUITY_LENGTH_CR'].transform('min')
combine['ANNUITY_LEN_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['ANNUITY_LENGTH_CR'].transform('min')


combine['EMPLOYED_m_BIRTH_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('min')
combine['EMPLOYED_m_BIRTH_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('min')


combine['app_EXT_mean_mean_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_mean_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_max_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_max_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_min_OCCUPATION_TYPE'] = combine.groupby('OCCUPATION_TYPE')['app EXT_SOURCE mean'].transform('min')
combine['app_EXT_mean_min_ORGANIZATION_TYPE'] = combine.groupby('ORGANIZATION_TYPE')['app EXT_SOURCE mean'].transform('min')

### Groupbyで作る特徴量２軍（FAMILYとEDUCATIONとHOUSING）

combine['INCOME_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].transform('min')
combine['INCOME_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_INCOME_TOTAL'].transform('min')
combine['INCOME_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_INCOME_TOTAL'].transform('min')


combine['CREDIT_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_CREDIT'].transform('mean')
combine['CREDIT_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_CREDIT'].transform('mean')
combine['CREDIT_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_CREDIT'].transform('mean')
combine['CREDIT_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_CREDIT'].transform('max')
combine['CREDIT_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_CREDIT'].transform('max')
combine['CREDIT_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_CREDIT'].transform('max')
combine['CREDIT_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_CREDIT'].transform('min')
combine['CREDIT_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_CREDIT'].transform('min')
combine['CREDIT_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_CREDIT'].transform('min')

combine['ANNUITY_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_ANNUITY'].transform('min')
combine['ANNUITY_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_ANNUITY'].transform('min')
combine['ANNUITY_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_ANNUITY'].transform('min')

combine['GOODS_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['AMT_GOODS_PRICE'].transform('min')
combine['GOODS_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['AMT_GOODS_PRICE'].transform('min')
combine['GOODS_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['AMT_GOODS_PRICE'].transform('min')

combine['EXT_SOURCE_1_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_1_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_1_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_1_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_1'].transform('min')
combine['EXT_SOURCE_1_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_1'].transform('min')
combine['EXT_SOURCE_1_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_1'].transform('min')

combine['EXT_SOURCE_2_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_2_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_2_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_2_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_2'].transform('min')
combine['EXT_SOURCE_2_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_2'].transform('min')
combine['EXT_SOURCE_2_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_2'].transform('min')

combine['EXT_SOURCE_3_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_3'].transform('mean')
combine['EXT_SOURCE_3_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_3'].transform('mean')
combine['EXT_SOURCE_3_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_3'].transform('mean')
combine['EXT_SOURCE_3_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EXT_SOURCE_3'].transform('min')
combine['EXT_SOURCE_3_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EXT_SOURCE_3'].transform('min')
combine['EXT_SOURCE_3_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EXT_SOURCE_3'].transform('min')

combine['ANNUITY_LEN_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['ANNUITY_LENGTH_CR'].transform('min')
combine['ANNUITY_LEN_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['ANNUITY_LENGTH_CR'].transform('min')
combine['ANNUITY_LEN_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['ANNUITY_LENGTH_CR'].transform('min')

combine['EMPLOYED_m_BIRTH_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['EMPLOYED_BIRTH_mainas'].transform('min')
combine['EMPLOYED_m_BIRTH_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['EMPLOYED_BIRTH_mainas'].transform('min')
combine['EMPLOYED_m_BIRTH_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['EMPLOYED_BIRTH_mainas'].transform('min')

combine['app_EXT_mean_mean_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_mean_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_mean_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_max_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_max_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_max_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_min_NAME_EDUCATION_TYPE'] = combine.groupby('NAME_EDUCATION_TYPE')['app EXT_SOURCE mean'].transform('min')
combine['app_EXT_mean_min_NAME_FAMILY_STATUS'] = combine.groupby('NAME_FAMILY_STATUS')['app EXT_SOURCE mean'].transform('min')
combine['app_EXT_mean_min_NAME_HOUSING_TYPE'] = combine.groupby('NAME_HOUSING_TYPE')['app EXT_SOURCE mean'].transform('min')

### Groupbyで作る特徴量3軍（GENDERとINCOME_TYPEとCONTRACT）
combine['INCOME_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].transform('mean')
combine['INCOME_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].transform('max')
combine['INCOME_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_INCOME_TOTAL'].transform('min')
combine['INCOME_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].transform('min')
combine['INCOME_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].transform('min')


combine['CREDIT_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_CREDIT'].transform('mean')
combine['CREDIT_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_CREDIT'].transform('mean')
combine['CREDIT_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_CREDIT'].transform('mean')
combine['CREDIT_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_CREDIT'].transform('max')
combine['CREDIT_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_CREDIT'].transform('max')
combine['CREDIT_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_CREDIT'].transform('max')
combine['CREDIT_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_CREDIT'].transform('min')
combine['CREDIT_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_CREDIT'].transform('min')
combine['CREDIT_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_CREDIT'].transform('min')

combine['ANNUITY_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_ANNUITY'].transform('mean')
combine['ANNUITY_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_ANNUITY'].transform('max')
combine['ANNUITY_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_ANNUITY'].transform('min')
combine['ANNUITY_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_ANNUITY'].transform('min')
combine['ANNUITY_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_ANNUITY'].transform('min')

combine['GOODS_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_GOODS_PRICE'].transform('mean')
combine['GOODS_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_GOODS_PRICE'].transform('max')
combine['GOODS_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['AMT_GOODS_PRICE'].transform('min')
combine['GOODS_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['AMT_GOODS_PRICE'].transform('min')
combine['GOODS_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['AMT_GOODS_PRICE'].transform('min')

combine['EXT_SOURCE_1_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_1_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_1_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_1'].transform('max')
combine['EXT_SOURCE_1_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_1'].transform('min')
combine['EXT_SOURCE_1_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_1'].transform('min')
combine['EXT_SOURCE_1_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_1'].transform('min')

combine['EXT_SOURCE_2_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_2_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_2_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_2'].transform('max')
combine['EXT_SOURCE_2_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_2'].transform('min')
combine['EXT_SOURCE_2_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_2'].transform('min')
combine['EXT_SOURCE_2_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_2'].transform('min')

combine['EXT_SOURCE_3_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_3'].transform('mean')
combine['EXT_SOURCE_3_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_3'].transform('mean')
combine['EXT_SOURCE_3_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_3'].transform('max')
combine['EXT_SOURCE_3_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_3'].transform('min')
combine['EXT_SOURCE_3_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EXT_SOURCE_3'].transform('min')
combine['EXT_SOURCE_3_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EXT_SOURCE_3'].transform('min')

combine['EXT_SOURCE_1_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_1'].transform('mean')
combine['EXT_SOURCE_2_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_2'].transform('mean')
combine['EXT_SOURCE_3_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EXT_SOURCE_3'].transform('mean')

combine['ANNUITY_LEN_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['ANNUITY_LENGTH_CR'].transform('mean')
combine['ANNUITY_LEN_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['ANNUITY_LENGTH_CR'].transform('max')
combine['ANNUITY_LEN_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['ANNUITY_LENGTH_CR'].transform('min')
combine['ANNUITY_LEN_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['ANNUITY_LENGTH_CR'].transform('min')
combine['ANNUITY_LEN_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['ANNUITY_LENGTH_CR'].transform('min')

combine['EMPLOYED_m_BIRTH_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EMPLOYED_BIRTH_mainas'].transform('mean')
combine['EMPLOYED_m_BIRTH_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EMPLOYED_BIRTH_mainas'].transform('max')
combine['EMPLOYED_m_BIRTH_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['EMPLOYED_BIRTH_mainas'].transform('min')
combine['EMPLOYED_m_BIRTH_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['EMPLOYED_BIRTH_mainas'].transform('min')
combine['EMPLOYED_m_BIRTH_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['EMPLOYED_BIRTH_mainas'].transform('min')

combine['app_EXT_mean_mean_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_mean_CODE_GENDER'] = combine.groupby('CODE_GENDER')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_mean_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['app EXT_SOURCE mean'].transform('mean')
combine['app_EXT_mean_max_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_max_CODE_GENDER'] = combine.groupby('CODE_GENDER')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_max_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['app EXT_SOURCE mean'].transform('max')
combine['app_EXT_mean_min_NAME_CONTRACT_TYPE'] = combine.groupby('NAME_CONTRACT_TYPE')['app EXT_SOURCE mean'].transform('min')
combine['app_EXT_mean_min_CODE_GENDER'] = combine.groupby('CODE_GENDER')['app EXT_SOURCE mean'].transform('min')
combine['app_EXT_mean_min_NAME_INCOME_TYPE'] = combine.groupby('NAME_INCOME_TYPE')['app EXT_SOURCE mean'].transform('min')

### 相関の高い特徴量を消去する
del combine["YEAR_BIRTH"]
del combine["YEAR_EMPLOYED"]
del combine["YEAR_REGISTRATION"]
del combine["YEAR_ID_PUBLISH"]
missing_data(combine).head(3)

### LableEncoding
from sklearn import preprocessing
for column in ["NAME_CONTRACT_TYPE", "CODE_GENDER"]:
    le = preprocessing.LabelEncoder()
    le.fit(combine[column])
    combine[column] = le.transform(combine[column])

### One-Hot-Encoding
categorical = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
               "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE", "NAME_EDUCATION_TYPE"]
combine = pd.get_dummies(combine, columns=categorical)
combine.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in combine.columns]

### Split Data
train = combine[combine["train"]==1]
test = combine[combine["train"]==0]

### train
X = train.drop(['TARGET', 'SK_ID_CURR', "train"], axis = 1)
y = train['TARGET']

### test
test_X = test.drop(['TARGET', 'SK_ID_CURR', "train"], axis = 1)
test_id = test['SK_ID_CURR']

### 全体の行列・列数
X.shape

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss

### 第一層用
# lightGBM用_gbdt（浅い）
# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv_gbdt(train_x, train_y, test_x, seed, depth):
    preds = []
    preds_test = []
    va_idxes = []

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(skf.split(train_x, train_y)):
        tr_x, va_x = train_x.iloc[tr_idx, :], train_x.iloc[va_idx, :]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)

        lgb_params = {'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': 'auc',
                      'learning_rate': 0.1,
                      'num_iterations': 1000,
                      'max_depth': depth,
                      'min_data_in_leaf': 20,
                      'feature_fraction': 0.8,
                      'bagging_fraction': 1.0,
                      'bagging_freq': 0,
                      'num_leaves': 2**8-1,
                      'max_bin': 200,
                      'boost_from_average': False,
                      'verbosity': -1,
                      'reg_alpha': 4.5,
                      'reg_lambda': 1.0,
                      'scale_pos_weight': 2} 
        
        model = lgb.train(lgb_params, lgb_train,
                        valid_sets=[lgb_train, lgb_eval], 
                        valid_names=['train', 'valid'],
                        #verbose_eval=False, # この行を削除またはコメントアウト
                        num_boost_round=100, 
                        #early_stopping_rounds=100
                        )
        
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test

### 第一層用
# lightGBM用_dart（浅い）
# 学習データに対する「目的変数を知らない」予測値と、テストデータに対する予測値を返す関数
def predict_cv_dart(train_x, train_y, test_x, seed, depth):
    preds = []
    preds_test = []
    va_idxes = []

    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)

    # クロスバリデーションで学習・予測を行い、予測値とインデックスを保存する
    for i, (tr_idx, va_idx) in enumerate(skf.split(train_x, train_y)):
        tr_x, va_x = train_x.iloc[tr_idx, :], train_x.iloc[va_idx, :]
        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)

        lgb_params = {'boosting_type': 'gbdt',
                      'objective': 'binary',
                      'metric': 'auc',
                      'learning_rate': 0.1,
                      'num_iterations': 1000,
                      'max_depth': depth,
                      'min_data_in_leaf': 20,
                      'feature_fraction': 0.8,
                      'bagging_fraction': 1.0,
                      'bagging_freq': 1,
                      'num_leaves': 2**8-1,
                      'max_bin': 200,
                      'boost_from_average': False,
                      'verbosity': -1,
                      'reg_alpha': 4.0,
                      'reg_lambda': 2,
                      'scale_pos_weight': 2} 
        
        model = lgb.train(lgb_params, lgb_train,
                        valid_sets=[lgb_train, lgb_eval], 
                        valid_names=['train', 'valid'],
                        #verbose_eval=False, # この行を削除またはコメントアウト
                        num_boost_round=100, 
                        #early_stopping_rounds=100
                        )
        
        pred = model.predict(va_x)
        preds.append(pred)
        pred_test = model.predict(test_x)
        preds_test.append(pred_test)
        va_idxes.append(va_idx)

    # バリデーションデータに対する予測値を連結し、その後元の順序に並べ直す
    va_idxes = np.concatenate(va_idxes)
    preds = np.concatenate(preds, axis=0)
    order = np.argsort(va_idxes)
    pred_train = preds[order]

    # テストデータに対する予測値の平均をとる
    preds_test = np.mean(preds_test, axis=0)

    return pred_train, preds_test

### lightGBM, boosting_type=gbdtでdepthとseedを変えて実装
# gbdt, depth=3
pred_train_1ag, pred_test_1ag = predict_cv_gbdt(X, y, test_X, 9, 3)
pred_train_2ag, pred_test_2ag = predict_cv_gbdt(X, y, test_X, 8, 3)
# gbdt, depth=3
pred_train_11ag, pred_test_11ag = predict_cv_gbdt(X, y, test_X, 29, 3)
pred_train_12ag, pred_test_12ag = predict_cv_gbdt(X, y, test_X, 28, 3)
# gbdt, depth=4
pred_train_21ag, pred_test_21ag = predict_cv_gbdt(X, y, test_X, 39, 4)
pred_train_22ag, pred_test_22ag = predict_cv_gbdt(X, y, test_X, 38, 4)
# gbdt, depth=5
pred_train_3bg, pred_test_3bg = predict_cv_gbdt(X, y, test_X, 5, 5)
pred_train_4bg, pred_test_4bg = predict_cv_gbdt(X, y, test_X, 4, 5)
# gbdt, depth=8
pred_train_5cg, pred_test_5cg = predict_cv_gbdt(X, y, test_X, 15, 8)
pred_train_6cg, pred_test_6cg = predict_cv_gbdt(X, y, test_X, 14, 8)

print("auc:", round(roc_auc_score(y, pred_train_1ag), 5))
print("auc:", round(roc_auc_score(y, pred_train_2ag), 5))
print("*"*6)
print("auc:", round(roc_auc_score(y, pred_train_11ag), 5))
print("auc:", round(roc_auc_score(y, pred_train_12ag), 5))
print("*"*6)
print("auc:", round(roc_auc_score(y, pred_train_21ag), 5))
print("auc:", round(roc_auc_score(y, pred_train_22ag), 5))
print("*"*6)
print("auc:", round(roc_auc_score(y, pred_train_3bg), 5))
print("auc:", round(roc_auc_score(y, pred_train_4bg), 5))
print("*"*6)
print("auc:", round(roc_auc_score(y, pred_train_5cg), 5))
print("auc:", round(roc_auc_score(y, pred_train_6cg), 5))
print("*"*6)

### 第一層の予測値を出す/各モデルの検証をする
# dart, depth=3
pred_train_1ad, pred_test_1ad = predict_cv_dart(X, y, test_X, 7, 3)
pred_train_2ad, pred_test_2ad = predict_cv_dart(X, y, test_X, 6, 3)
# dart, depth=3
pred_train_11ad, pred_test_11ad = predict_cv_dart(X, y, test_X, 17, 3)
pred_train_12ad, pred_test_12ad = predict_cv_dart(X, y, test_X, 16, 3)
# dart, depth=4
pred_train_21ad, pred_test_21ad = predict_cv_dart(X, y, test_X, 27, 4)
pred_train_22ad, pred_test_22ad = predict_cv_dart(X, y, test_X, 26, 4)

print("auc:", round(roc_auc_score(y, pred_train_1ad), 5))
print("auc:", round(roc_auc_score(y, pred_train_2ad), 5))
print("*"*6)
print("auc:", round(roc_auc_score(y, pred_train_11ad), 5))
print("auc:", round(roc_auc_score(y, pred_train_12ad), 5))
print("*"*6)
print("auc:", round(roc_auc_score(y, pred_train_21ad), 5))
print("auc:", round(roc_auc_score(y, pred_train_22ad), 5))
print("*"*6)

### 予測値を特徴量としてデータフレームを作成
train_x_2 = pd.DataFrame({'pred_1ag': pred_train_1ag, 'pred_2ag': pred_train_2ag, 'pred_1ad': pred_train_1ad, 'pred_2ad': pred_train_2ad,
                          'pred_11ag': pred_train_11ag, 'pred_12ag': pred_train_12ag, 'pred_11ad': pred_train_11ad, 'pred_12ad': pred_train_12ad,
                          'pred_21ag': pred_train_21ag, 'pred_22ag': pred_train_22ag, 'pred_21ad': pred_train_21ad, 'pred_22ad': pred_train_22ad,
                          'pred_3bg': pred_train_3bg, 'pred_4bg': pred_train_4bg,'pred_5cg': pred_train_5cg, 'pred_6cg': pred_train_6cg})
test_x_2 = pd.DataFrame({'pred_1ag': pred_test_1ag, 'pred_2ag': pred_test_2ag, 'pred_1ad': pred_test_1ad, 'pred_2ad': pred_test_2ad,
                         'pred_11ag': pred_test_11ag, 'pred_12ag': pred_test_12ag, 'pred_11ad': pred_test_11ad, 'pred_12ad': pred_test_12ad,
                         'pred_21ag': pred_test_21ag, 'pred_22ag': pred_test_22ag, 'pred_21ad': pred_test_21ad, 'pred_22ad': pred_test_22ad,
                         'pred_3bg': pred_test_3bg, 'pred_4bg': pred_test_4bg, 'pred_5cg': pred_test_5cg, 'pred_6cg': pred_test_6cg})

### 単純平均を行う
train_x_2["mean"] = train_x_2.mean(axis=1)
test_x_2["mean"] = test_x_2.mean(axis=1)

### modelのvalidationスコアを出す
print("auc:", round(roc_auc_score(y, train_x_2["mean"]), 5))

submission = pd.DataFrame({
        "SK_ID_CURR": test_id,
        "TARGET":test_x_2["mean"]
    })
submission.to_csv('submission.homecredit_of_the_best.csv', index=False)