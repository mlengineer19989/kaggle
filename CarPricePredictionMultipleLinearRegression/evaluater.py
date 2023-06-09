import numpy as np
import pandas as pd
import typing as tp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from dataclasses import dataclass

@dataclass
class LNmodel():
    X:pd.DataFrame              #選択された特徴量
    y:pd.Series                 #目的変数
    model:LinearRegression      #選択された特徴量を用いて生成された線形回帰オブジェクト

    @property
    def n_sample(self) -> int:
        return self.X.shape[0]
    
    @property
    def n_feature(self) -> int:
        return self.X.shape[1]

    @property
    def Se(self):
        """残差平方和
        """
        return self.n_sample * mean_squared_error(self.y, self.model.predict(self.X))
    
    @property
    def Sy(self):
        return self.n_sample * np.var(self.y)
    
    @property
    def phi_T(self):
        return self.n_sample-1
    
    @property
    def phi_e(self):
        return self.n_sample-self.n_feature-1
    
    @property
    def sigma(self):
        return self.Se/self.phi_e
    
    @property
    def R2(self):
        return r2_score(self.y, self.model.predict(self.X))
    
    @property
    def adj_R2(self):
        return 1-(self.Se/self.phi_e)/(self.Sy/self.phi_T)
    
    @staticmethod
    def generate_LNmodel(X:pd.DataFrame, y:pd.Series) -> "LNmodel":
        model = LinearRegression()
        model.fit(X, y)
        return LNmodel(X=X, y=y, model=model)

class FeatureSelector():
    def __init__(self, df_preprocessed:pd.DataFrame, df_y:pd.Series) -> None:
        self.df_preprocessed:pd.DataFrame = df_preprocessed
        self.df_y:pd.Series = df_y

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df_preprocessed, self.df_y, test_size=0.3)

    def get_sorted_feature_labels(self) -> tp.List[str]:
        #すべての特徴量を使って回帰を行う。ここで得た偏回帰係数の絶対値が大きい順に変数増加法を行う
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        return self.X_train.columns[np.argsort(abs(model.coef_))]

    def select_feature(self) -> tp.List[str]:
        selected_feature_label:tp.List[str] = []
        m_current:LNmodel = None

        # TODO :変数の取り出しロジックは質的変数の場合に向けて変更する必要がある。
        for i, feature in enumerate(self.get_sorted_feature_labels()):
            #それまで選択された特徴量に、今のループの特徴量を追加してモデルを作成する。
            m_next:LNmodel = LNmodel.generate_LNmodel(X=self.X_train[selected_feature_label+[feature]], y=self.y_train)

            #最初の特徴量は必ず加える。
            if i == 0:
                selected_feature_label.append(feature)
                m_current = m_next
                continue

            #現モデルと次のモデルの比較をF値で行う。
            F:np.float64 = FeatureSelector.F(m_current=m_current, m_next=m_next)
            if F>2:
                selected_feature_label.append(feature)
                m_current = m_next
        return selected_feature_label
    
    @staticmethod
    def F(m_current:LNmodel, m_next:LNmodel) -> np.float64:
        nume = (m_current.Se-m_next.Se)/(m_current.phi_e-m_next.phi_e)
        deno = m_next.Se/m_next.phi_e
        return nume/deno