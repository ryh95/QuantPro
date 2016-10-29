# First XGBoost model for Pima Indians dataset
import numpy
import xgboost
from matplotlib import pyplot
from sklearn import cross_validation
from sklearn.metrics import r2_score
# load data
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from xgboost import plot_importance

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# rescale X
from sklearn.preprocessing import minmax_scale
minmax_scale(X,copy=False)


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=seed)


###########################################
# try some models e.g. XGBoost

# fit model no training data
model = xgboost.XGBClassifier()

eval_set = [(X_test,y_test)]
model.fit(X_train, y_train,eval_metric="logloss", eval_set=eval_set, verbose=True,early_stopping_rounds=10)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = r2_score(y_test, predictions)
print("Accuracy: %.2f" % (accuracy * 100.0))

###########################################
#Now let's Learn some other models...
#benchmark model is OLS

from sklearn import linear_model
from sklearn.metrics import r2_score,mean_squared_error

ols = linear_model.LinearRegression()
pred_ols = ols.fit(X_train,y_train).predict(X_test)
print("Accuracy: %.2f (for OLS)" % (r2_score(y_test,pred_ols) * 100.0))

###########################################
# plot feature importance
plot_importance(model)
pyplot.show()

###########################################
# model tuning
model = xgboost.XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="r2", n_jobs=-1, cv=kfold)
result = grid_search.fit(X, Y)
# summarize results
print("Best: %f using %s" % (result.best_score_, result.best_params_))