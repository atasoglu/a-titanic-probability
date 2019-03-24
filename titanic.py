# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('titanic.csv')

data = data.drop(labels=['Name'], axis=1)

Y = data.iloc[:,0:1]

X = data.iloc[:,1:]

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
# import statsmodels.api as sm
from sklearn.metrics import r2_score

sc = StandardScaler()

ohe = OneHotEncoder(categorical_features='all')

sex = X.iloc[:,1:2]

sex = ohe.fit_transform(sex).toarray()

sex = pd.DataFrame(data = sc.fit_transform(sex),
                   index = range(887),
                   columns=['f','m'])

X = X.drop(labels=['Sex'], axis=1)
X = pd.DataFrame(data = sc.fit_transform(X), index=range(887),
                 columns=['plcass','age','sib/spo','par/child','fare'])
X = pd.concat([X, sex], axis=1)

Y = pd.DataFrame(data = sc.fit_transform(Y), index=range(887), columns=['survived'])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.33,
        random_state=0)

from sklearn.svm import SVR

svr = SVR(kernel='rbf')

svr.fit(x_train, y_train)

y_pred = svr.predict(x_test)

r2 = r2_score(y_test, y_pred)

print('r2 score:',r2)


