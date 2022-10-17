import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#invite people for the Kaggle party
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

def preprocess_for_xgb(train, test):
    """以下のサイトで実装されている勾配ブースティング決定木用の前処理
    https://www.kaggle.com/code/anandhuh/house-price-prediction-simple-solution-top-3

    Args:
        train (_type_): _description_
        test (_type_): _description_

    Returns:
        _type_: _description_
    """

    # Remove rows with missing target
    X = train.dropna(axis=0, subset=['SalePrice'])

    # separate target from predictors
    y = X.SalePrice              
    X.drop(['SalePrice'], axis=1, inplace=True)

    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y,
                                                                    train_size=0.8,
                                                                    test_size=0.2,
                                                                    random_state=0)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)

    low_cardinality_cols = [cname for cname in X_train_full.columns 
                            if X_train_full[cname].nunique() < 10 and 
                            X_train_full[cname].dtype == "object"]

    # Select numeric columns
    numeric_cols = [cname for cname in X_train_full.columns
                    if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = low_cardinality_cols + numeric_cols

    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    # for test data also
    X_test = test[my_cols].copy()

    # One-hot encode the data
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)

    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    return X_train, y_train, X_valid, y_valid, X_test

def preprocess_selfmade(train, test_data, y_test):
    """ネットを参考に自分で実装した前処理の流れ

    Args:
        train (dataframe): train data involving both X and y.
        test_data (dataframe): test data involving only X.
        y_test (pd.series): test data involving only y.

    Returns:
        X_train : _description_
        y_train (pd.series) :
        X_test
        y_test (pd.series) : 
        
    """
    
    #訓練データの各変数間の相関行列の計算
    corrmat = train.corr()
    selected_label, distmat, row_clusters = var_select_by_clustring(corrmat, "SalePrice")
    selected_label = highcorr_compare(selected_label, distmat, corrmat, "SalePrice")

    #欠損値処理
    missing_data_train = missing_data_analysis(train)
    train = train.drop((missing_data_train[missing_data_train['Total'] > 1]).index,1)
    train = train.drop(train.loc[train['Electrical'].isnull()].index)


    test_data = test_data.drop((missing_data_train[missing_data_train['Total'] > 1]).index,1)
    test_data = test_data.drop(test_data.loc[test_data['Electrical'].isnull()].index)
    missing_data_test = missing_data_analysis(test_data)
    test_data=test_data.fillna(test_data.mean())
    test_data = test_data.fillna(method="ffill")

    #外れ値削除
    train.sort_values(by = 'GrLivArea', ascending = False)[:2]
    train = train.drop(train[train['Id'] == 1299].index)
    train = train.drop(train[train['Id'] == 524].index)

    #対数変換
    train['SalePrice'] = np.log(train['SalePrice'])
    train['GrLivArea'] = np.log(train['GrLivArea'])
    test_data['GrLivArea'] = np.log(test_data['GrLivArea'])

    #create column for new variable (one is enough because it's a binary categorical feature)
    #if area>0 it gets 1, for area==0 it gets 0
    train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
    train['HasBsmt'] = 0 
    train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1

    test_data['HasBsmt'] = pd.Series(len(test_data['TotalBsmtSF']), index=test_data.index)
    test_data['HasBsmt'] = 0 
    test_data.loc[test_data['TotalBsmtSF']>0,'HasBsmt'] = 1

    #transform data
    train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
    test_data.loc[test_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(test_data['TotalBsmtSF'])

    #目的変数、説明変数の設定
    X_train = train.drop(["SalePrice"], axis=1)
    y_train = train["SalePrice"]


    X_test = test_data
    y_test = np.log(y_test)  #np.log(pd.read_csv(os.path.join(data_dirname, "sample_submission.csv"))["SalePrice"])
    
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    X_total = pd.concat([X_train, X_test], axis=0)

    #クラスタリング結果を反映する
    for l in selected_label:
        if l not in X_total.columns:
            print(l)
            selected_label.remove(l)
    X_total=X_total.loc[:,selected_label]

    #カテゴリ変数を変換
    X_total = pd.get_dummies(X_total)

    X_train = X_total[:train_size]
    X_test = X_total[train_size:train_size+test_size]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                                    train_size=0.8,
                                                                    test_size=0.2,
                                                                    random_state=0)

    return X_train, y_train, X_valid, y_valid, X_test, y_test



def var_select_by_clustring(corrmat, y_label):
    """相関係数行列に基づいて、以下の処理を行う。
    ・目的変数との相関が低い変数の削除
    ・説明変数同士の非類似度行列の作成
    ・階層型クラスタリングによるlinkageの作成

    Args:
        corrmat (dataframe): _description_
        y_label (str): _description_

    Returns:
        _type_: _description_
    """
    selected_label = corrmat[y_label].abs()[corrmat[y_label].abs() > 0.3].index.to_list()
    selected_corrmat = corrmat.loc[selected_label, selected_label].drop(index=y_label).drop(columns=y_label)
    distmat = 1- selected_corrmat

    row_clusters = linkage(squareform(distmat), method="complete", metric="euclidean")

    return selected_label, distmat, row_clusters



def highcorr_compare(selected_label, distmat, corrmat, y_label):
    """クラスタリングで同じクラスに分類されたものから１つを選択する。

    Args:
        selected_label (list): _description_
        distmat (dataframe): _description_
        y_label (str): _description_

    Returns:
        _type_: _description_
    """
    pair_list = list(np.where(distmat[:]<0.3))
    highcorrpair = []
    for i in range(len(pair_list[0])):
        ind_num = pair_list[0][i]
        col_num = pair_list[1][i]
        if ind_num < col_num:
            highcorrpair.append([distmat.index[ind_num], distmat.index[col_num]])

            if corrmat[y_label][ind_num] < corrmat[y_label][col_num]:
                selected_label.remove(selected_label[ind_num])
            elif corrmat[y_label][ind_num] > corrmat[y_label][col_num]:
                selected_label.remove(selected_label[col_num])

    return selected_label


def missing_data_analysis(data):
    """dataの欠損値情報データフレームを作成する。

    Args:
        data (dataframe): _description_

    Returns:
        dataframe: _description_
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
