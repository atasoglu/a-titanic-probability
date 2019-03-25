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

# method1: svr 

from sklearn.svm import SVR

svr = SVR(kernel='rbf')
svr.fit(x_train, y_train)
y_pred = svr.predict(x_test)

r2 = r2_score(y_test, y_pred)

print('svr r2:',r2)

# method2: multiple linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

r2 = r2_score(y_test, y_pred)

print('mlr r2:',r2)

# method3: polynomial regression

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)

poly_x_train = poly_reg.fit_transform(x_train)
poly_x_test = poly_reg.fit_transform(x_test)

poly_r = LinearRegression()
poly_r.fit(poly_x_train, y_train)

y_pred = poly_r.predict(poly_x_test)

r2 = r2_score(y_test, y_pred)

print('polynomial reg r2:',r2)

# method4: decision trees

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=0)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

r2 = r2_score(y_test, y_pred)

print('decision trees r2',r2)

# method5: random forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=10, random_state=0)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

r2 = r2_score(y_test, y_pred)

print('random forest reg:', r2)

# plt graphs:
# plt.scatter(range(293),y_test, color='red')
# plt.scatter(range(293),y_pred, color='blue')
# plt.title(label='regression type')
# plt.legend(['true', 'predict'])
# plt.show()

