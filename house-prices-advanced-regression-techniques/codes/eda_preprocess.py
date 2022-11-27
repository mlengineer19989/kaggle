import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

#invite people for the Kaggle party
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import torch

from scipy import stats

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from hyperopt import hp
from keras.callbacks import EarlyStopping
from keras.layers import ReLU, PReLU
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler

def reset_data(data_dirname):
    train = pd.read_csv(os.path.join(data_dirname, "train.csv"))
    test_data = pd.read_csv(os.path.join(data_dirname, "test.csv"))
    y_test = pd.read_csv(os.path.join(data_dirname, "sample_submission.csv"))["SalePrice"]
    return train, test_data, y_test

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

def preprocess_selfmade(train, test_data, y_test, validation=True):
    """ネットを参考に自分で実装した前処理の流れ

    Args:
        train (dataframe): train data involving both X and y.
        test_data (dataframe): test data involving only X.
        y_test (pd.series): test data involving only y.
        validation (bool): train_test_split to obtain train data and validation data is done if this param is true.

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
    # for l in selected_label:
    #     if l not in X_total.columns:
    #         print(l)
    #         selected_label.remove(l)
    # X_total=X_total.loc[:,selected_label]

    #カテゴリ変数を変換
    X_total = pd.get_dummies(X_total)

    X_train = X_total[:train_size]
    X_test = X_total[train_size:train_size+test_size]

    if validation:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                                        train_size=0.8,
                                                                        test_size=0.2,
                                                                        random_state=0)
    else:
        X_valid = None
        y_valid = None

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
        dataframe: _description
    """
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data


def calc_score(y_true, y_pred):
    MSE = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return MSE, r2

def lin_regplot(y, y_pred):
    plt.scatter(y, y_pred, c="steelblue", edgecolors="white", s=70)
    plt.plot(y, y, color="black", lw=2)
    plt.xlabel("y_true")
    plt.ylabel("y_predict")

class evaluation_show():
    def __init__(self, X_train, y_train, X_test, y_test, model) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.model = model

    def predict_with_model(self, X):
        return self.model.predict(X)

    def predict_train_and_test(self):
        self.y_pred_train = self.predict_with_model(self.X_train)
        self.y_pred_test = self.predict_with_model(self.X_test)
        
    def show_result(self):
        self.predict_train_and_test()
        MSE_train, r2_train = calc_score(self.y_train, self.y_pred_train)
        MSE_test, r2_test = calc_score(self.y_test, self.y_pred_test)

        print("RMSE_train={0}, r2_train={1}".format(MSE_train, r2_train))
        print("RMSE_test={0}, r2_test={1}".format(MSE_test, r2_test))

        lin_regplot(self.y_train, self.y_pred_train)
        plt.title("predict of train data")
        plt.show()

        lin_regplot(self.y_test, self.y_pred_test)
        plt.title("predict of test data")
        plt.show()

class evaluation_show_pytorch(evaluation_show):
    def predict_with_model(self, X):
        self.model.eval()
        X_valid_for_torch = torch.from_numpy(X.values).float()
        with torch.no_grad():
            y_pred_tensor = self.model(X_valid_for_torch)

        y_pred = y_pred_tensor.data.numpy()
        y_pred = pd.Series(data=y_pred.squeeze())

        return y_pred

class MLP():

    def __init__(self, params):
        self.params = params
        self.scaler = None
        self.model = None
    
    
    def fit(self, tr_x, tr_y, va_x, va_y):

        # パラメータ
        input_dropout = self.params['input_dropout']
        hidden_layers = int(self.params['hidden_layers'])
        hidden_units = int(self.params['hidden_units'])
        hidden_activation = self.params['hidden_activation']
        hidden_dropout = self.params['hidden_dropout']
        batch_norm = self.params['batch_norm']
        optimizer_type = self.params['optimizer']['type']
        optimizer_lr = self.params['optimizer']['lr']
        batch_size = int(self.params['batch_size'])

        # 標準化
        self.scaler = StandardScaler()
        tr_x = self.scaler.fit_transform(tr_x)
        va_x = self.scaler.transform(va_x)

        self.model = Sequential()

        # 入力層
        self.model.add(Dropout(input_dropout, input_shape=(tr_x.shape[1],)))

        # 中間層
        for i in range(hidden_layers):
            self.model.add(Dense(hidden_units))
            if batch_norm == 'before_act':
                self.model.add(BatchNormalization())
            if hidden_activation == 'prelu':
                self.model.add(PReLU())
            elif hidden_activation == 'relu':
                self.model.add(ReLU())
            else:
                raise NotImplementedError
            self.model.add(Dropout(hidden_dropout))

        # 出力層
        self.model.add(Dense(1))

        # オプティマイザ
        if optimizer_type == 'sgd':
            optimizer = SGD(lr=optimizer_lr, decay=1e-6, momentum=0.9, nesterov=True)
        elif optimizer_type == 'adam':
            optimizer = Adam(lr=optimizer_lr, beta_1=0.9, beta_2=0.999, decay=0.)
        else:
            raise NotImplementedError

        # 目的関数、評価指標などの設定
        self.model.compile(loss='mse',
                           optimizer=optimizer, metrics=['mae', 'mse'])

        # エポック数、アーリーストッピング
        # あまりepochを大きくすると、小さい学習率のときに終わらないことがあるので注意
        nb_epoch = 200
        patience = 20
        early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

        # 学習の実行
        history = self.model.fit(tr_x, tr_y,
                                 epochs=nb_epoch,
                                 batch_size=batch_size, verbose=1,
                                 validation_data=(va_x, va_y),
                                 callbacks=[early_stopping])

        return history

    def predict(self, x):
        # 予測
        x = self.scaler.transform(x)
        y_pred = self.model.predict(x)
        y_pred = y_pred.flatten()
        return y_pred


def plot_history(hist):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [Thousand Dollars$^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    plt.ylim([0,50])