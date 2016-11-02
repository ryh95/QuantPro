# encoding:utf8
import numpy
import pandas as pd
import numpy as np
import xgboost
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from matplotlib import pyplot
from sklearn.metrics import r2_score

from xgboost import plot_importance

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
# model = xgboost.XGBRegressor(n_estimators=3000,learning_rate=0.1)
#
# eval_set = [(test_x,test_y)]
# model.fit(train_x, train_y,eval_metric="logloss", eval_set=eval_set, verbose=True,early_stopping_rounds=10)
# # make predictions for test data
# y_pred = model.predict(test_x)
# # evaluate predictions
# accuracy = r2_score(test_y, y_pred)
# print("Accuracy: %.2f" % (accuracy * 100.0))
#
# # Todo:plot learning curve
#
# # plot feature importance
# plot_importance(model)
# pyplot.show()

###########################################
#Now let's Learn some benchmark models...
#benchmark model is OLS

from sklearn import linear_model
# from sklearn.metrics import r2_score,mean_squared_error
#
# ols = linear_model.LinearRegression()
# pred_ols = ols.fit(train_x,train_y).predict(test_x)
# print("Accuracy: %.2f (for OLS)" % (r2_score(test_y,pred_ols) * 100.0))


###########################################
# try SVR(SVM)
# from sklearn import svm
# Todo:figure out why SVR takes a long long time to train
# svr = svm.SVR(kernel='linear')
# pred_svr = svr.fit(train_x,train_y).predict(test_x)
# print("Accuracy: %.2f (for SVR)" % (r2_score(test_y,pred_svr) * 100.0))

###########################################
# try ridge regression
# ridge = linear_model.RidgeCV(alphas=[11.0])
# pred_ridge = ridge.fit(train_x,train_y).predict(test_x)
# print("Accuracy: %.2f (for Ridge)" % (r2_score(test_y,pred_ridge) * 100.0))
# print ridge.alpha_

###########################################
# try LASSO
# figure out why r2 score is -0.00
# reason: the alpha is too big
# if assign a small value to alpha it would be fine e.g. 0.000085
# lasso = linear_model.Lasso(random_state=4,alpha=0.00009,max_iter=2000)
# pred_lasso = lasso.fit(train_x,train_y).predict(test_x)
# print("Accuracy: %.2f (for LASSO)" % (r2_score(test_y,pred_lasso) * 100.0))

###########################################
# try LASSOCV
# lasso = linear_model.LassoCV(n_jobs=4,random_state=4,n_alphas=4,alphas=[0.00009,0.000085],max_iter=2000)
# pred_lasso = lasso.fit(train_x,train_y).predict(test_x)
# print("Accuracy: %.2f (for LASSO)" % (r2_score(test_y,pred_lasso) * 100.0))
# print lasso.alpha_
# print lasso.n_iter_

###########################################
# try MLP
from keras.models import Sequential

# seed = 7
# numpy.random.seed(seed)
#
# model = Sequential()
# model.add(Dense(output_dim=512,input_shape=(332,)))
# model.add(Activation('tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(output_dim=256,input_shape=(332,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(output_dim=128,input_shape=(332,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(output_dim=64,input_shape=(332,)))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(output_dim=32))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.add(Activation('linear'))
# model.summary()
# model.compile(loss='mse',optimizer=RMSprop(),metrics=['mean_squared_error'])
# history = model.fit(train_x,train_y,batch_size=10,nb_epoch=30,verbose=1,validation_data=(test_x,test_y))
# pred_mlp = model.predict(test_x,verbose=1)
# model.save('my_model.h5')
# print("Accuracy: %.2f (for MLP)" % (r2_score(test_y,pred_mlp) * 100.0))

###########################################
#try random forest
from sklearn import ensemble
rand_forest = ensemble.RandomForestRegressor(n_estimators=3000,max_depth=2,random_state=4,n_jobs=4)
pred_rand_forest = rand_forest.fit(train_x,train_y).predict(test_x)
print("Accuracy: %.2f (for RandomForest)" % (r2_score(test_y,pred_rand_forest) * 100.0))