import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#invite people for the Kaggle party
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


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
