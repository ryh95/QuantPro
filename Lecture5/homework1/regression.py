import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Lecture5.homework1.model import get_tuned_svr, get_tuned_random_forest_regressor
from Lecture5.tools.plot_confusion import plot_confusion_matrix

df = pd.read_csv('hs300_dataset.csv')

df.index = pd.to_datetime(df['date'])
df.drop('date',axis=1,inplace=True)

# get close series
close = df['close'].copy()

# feature rescale
df = (df - df.mean()) / (df.max() - df.min())

# add next close as target value
df['target'] = df['close'].shift(-1)
df.dropna(inplace=True)


X = df.iloc[:,:-1]
y = df['target']
# split data set
threshold = 0.8

# Create training and test sets
X_train = X.iloc[:int(threshold*len(X)),:]
X_test = X.iloc[int(threshold*len(X)):,:]
y_train = y.iloc[:int(threshold*len(y))]
y_test = y.iloc[int(threshold*len(y)):]


# get best model using grid search
best_svc = get_tuned_svr(X_train, y_train)
# score the model
y_pred = best_svc.predict(X_test)
print best_svc
print r2_score(y_pred=y_pred, y_true=y_test)
# prices retrival
y_pred_retrival = y_pred*(close.max()-close.min())+close.mean()
y_test_retrival = y_test*(close.max()-close.min())+close.mean()

y_pred = pd.DataFrame(index=y_test_retrival.index)
y_pred['close'] = y_pred_retrival

# visulize the result
plt.figure(figsize=(9,5))
plt.plot(y_pred['close'],'r',lw=2,label='Predict Prices (red)')
plt.plot(y_test_retrival,'b',lw=2,label='True Prices (blue)')
plt.legend(loc=2,prop={'size':40})
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()


best_rf = get_tuned_random_forest_regressor(X_train, y_train)
# score the model
y_pred = best_rf.predict(X_test)
print best_rf
print r2_score(y_pred=y_pred, y_true=y_test)

# prices retrival
y_pred_retrival = y_pred*(close.max()-close.min())+close.mean()

y_pred = pd.DataFrame(index=y_test_retrival.index)
y_pred['close'] = y_pred_retrival


# visulize the result
plt.figure(figsize=(9,5))
plt.plot(y_pred['close'],'r',lw=2,label='Predict Prices (red)')
plt.plot(y_test_retrival,'b',lw=2,label='True Prices (blue)')
plt.legend(loc=2,prop={'size':40})
plt.grid(True)
plt.setp(plt.gca().get_xticklabels(), rotation=30)
plt.show()


# show feature importance
importances = best_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()