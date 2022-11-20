import numpy as np # linear algebra
import platform


import warnings
warnings.filterwarnings('ignore')

#import missingno as msno

import eda_preprocess as ep

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/Users/yamauchito_satoshi/Documents/data/kaggle/house-prices-advanced-regression-techniques'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


if platform.system()=="Windows":
    #windowsの場合
    data_dirname = r"C:\Users\yamau\OneDrive\ドキュメント\Python Scripts\AI\kaggle\house-prices-advanced-regression-techniques\data"
elif platform.system()=="Linux":
    #Linuxの場合
    data_dirname=r"./data"
else:    
    # macの場合
    data_dirname = r'/Users/yamauchito_satoshi/Documents/data/kaggle/house-prices-advanced-regression-techniques'

from sklearn.model_selection import train_test_split

from hyperopt import hp

from hyperopt import fmin, tpe, STATUS_OK, Trials
from sklearn.metrics import mean_squared_error

def score(params):
    # パラメータセットを指定したときに最小化すべき関数を指定する
    # モデルのパラメータ探索においては、モデルにパラメータを指定して学習・予測させた場合のスコアとする
    model = ep.MLP(params)
    model.fit(X_train_keras, y_train_keras, X_valid_keras, y_valid_keras)
    va_pred = model.predict(X_valid_keras)
    score = mean_squared_error(y_valid_keras, va_pred)
    print(f'params: {params}, logloss: {score:.4f}')

    # 情報を記録しておく
    history.append((params, score))

    return {'loss': score, 'status': STATUS_OK}

#自作前処理の場合
train, test_data, y_test = ep.reset_data(data_dirname)
X_train, y_train, X_valid, y_valid, X_test, y_test = ep.preprocess_selfmade(train, test_data, y_test)


param_space = {
    'input_dropout': hp.quniform('input_dropout', 0, 0.2, 0.05),
    'hidden_layers': hp.quniform('hidden_layers', 2, 4, 1),
    'hidden_units': hp.quniform('hidden_units', 32, 256, 32),
    'hidden_activation': hp.choice('hidden_activation', ['prelu', 'relu']),
    'hidden_dropout': hp.quniform('hidden_dropout', 0, 0.3, 0.05),
    'batch_norm': hp.choice('batch_norm', ['before_act']),
    'optimizer': hp.choice('optimizer',
                           [{'type': 'adam',
                             'lr': hp.loguniform('adam_lr', np.log(0.00001), np.log(0.01))},
                            {'type': 'sgd',
                             'lr': hp.loguniform('sgd_lr', np.log(0.00001), np.log(0.01))}]),
    'batch_size': hp.quniform('batch_size', 32, 128, 32),
}

X_train_keras, X_valid_keras, y_train_keras, y_valid_keras = train_test_split(X_train, y_train,
                                                                    train_size=0.8,
                                                                    test_size=0.2,
                                                                    random_state=0)

# hyperoptによるパラメータ探索の実行
max_evals = 10
trials = Trials()
history = []
fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 記録した情報からパラメータとスコアを出力する
# trialsからも情報が取得できるが、パラメータを取得しにくい
history = sorted(history, key=lambda tpl: tpl[1])
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')