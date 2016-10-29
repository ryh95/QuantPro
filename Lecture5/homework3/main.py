# encoding:utf8

import pandas as pd
import numpy as np
import xgboost
from sklearn.metrics import r2_score

from Lecture5.tools.plot_learning_curve import plot_learning_curve

df = pd.read_csv('./data/HFT_XY_unselected.csv') #从readme.txt中下载
col_names_x = df.columns[1:333]
x_all = np.asarray(df[col_names_x])
col_names_y = df.columns[333]
y_all = np.asarray(df[col_names_y])
print col_names_x
print col_names_y
print x_all.shape
print y_all.shape

# train test split
train_x = x_all[0:x_all.shape[0]*7/10]
train_y = y_all[0:y_all.shape[0]*7/10]
test_x = x_all[x_all.shape[0]*7/10:]
test_y = y_all[y_all.shape[0]*7/10:]
print train_y.shape
print train_x.shape
print test_x.shape
print test_y.shape

###########################################
# try some models e.g. XGBoost

# fit model no training data
model = xgboost.XGBRegressor(n_estimators=3000,learning_rate=0.07)

eval_set = [(test_x,test_y)]
model.fit(train_x, train_y,eval_metric="logloss", eval_set=eval_set, verbose=True,early_stopping_rounds=10)
# make predictions for test data
y_pred = model.predict(test_x)
# evaluate predictions
accuracy = r2_score(test_y, y_pred)
print("Accuracy: %.2f" % (accuracy * 100.0))

###########################################
# Todo:plot learning curve


###########################################
#Now let's Learn some benchmark models...
#benchmark model is OLS

from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error

ols = linear_model.LinearRegression()
pred_ols = ols.fit(train_x,train_y).predict(test_x)
print("Accuracy: %.2f (for OLS)" % (r2_score(test_y,pred_ols) * 100.0))