import platform
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import RidgeCV, Ridge

import eda_preprocess as ep

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




#obtain data and preprocess
train, test_data, y_test = ep.reset_data(data_dirname)
X_train, y_train, X_valid, y_valid, X_test, y_test = ep.preprocess_selfmade(train, test_data, y_test)

#クロスバリデーションにより、ハイパーパラメータ決定
slr = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
slr.fit(X_train, y_train)
print("alpha=", slr.alpha_)


#決定したalphaを使ってモデルを作成
train, test_data, y_test = ep.reset_data(data_dirname)
X_train, y_train, X_valid, y_valid, X_test, y_test = ep.preprocess_selfmade(train, test_data, y_test, validation=False)

lm = Ridge(alpha=slr.alpha_)
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

#評価値　RMSE計算
RMSE_test, r2_test = ep.calc_score(y_test, y_pred)
print("RMSE_test={0}, r2_test={1}".format(RMSE_test, r2_test))

#前処理で対数変換を行なっているため、もとに戻す。
y_pred = np.exp(y_pred)

#提出用csvファイル作成
submission = pd.DataFrame()
submission['Id']= test_data.Id
submission['SalePrice'] = y_pred
submission.to_csv(os.path.join(data_dirname, 'ridge_optimized.csv'), index = False)
