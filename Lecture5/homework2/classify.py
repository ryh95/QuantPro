
import tushare as ts
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from Lecture5.tools import prepare_dataset
from Lecture5.tools.model import get_tuned_svc, get_tuned_random_forest_classifier
from Lecture5.tools.plot_confusion import plot_confusion_matrix

######################################################
# get data first

# start,end = '2013-10-01','2016-10-28'
# df = ts.get_hist_data('600196',start=start,end=end)
# df.to_csv('raw_data.csv')

# df = pd.read_csv('raw_data.csv')
#
# df = prepare_dataset.create_features(df)
# df.to_csv('dataset.csv')
######################################################
# read in data add target
df = pd.read_csv('./data/dataset.csv')

df.index = pd.to_datetime(df['date'])
df.drop('date',axis=1,inplace=True)

# feature rescale
df = (df - df.mean()) / (df.max() - df.min())

# add direction as target value
df['direction'] = np.sign(df['return'])
df['direction'] = df['direction'].shift(-1)
df.dropna(inplace=True)

######################################################
# train test split
X = df.iloc[:,:-1]
y = df['direction']
# split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

######################################################
# using SVC to classify

# get best model using grid search
best_svc = get_tuned_svc(X_train, y_train)

# score the model
y_pred = best_svc.predict(X_test)
print best_svc
print r2_score(y_pred=y_pred, y_true=y_test)

# visulize the result
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[-1.0, 1.0])

# print cnf_matrix
classes = [-1.0, 1.0]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes, title='Confusion matrix, without normalization')
plt.show()

######################################################
# using RandomForestClassifier

# get best model using grid search
best_rf = get_tuned_random_forest_classifier(X_train, y_train)

# score the model
y_pred = best_rf.predict(X_test)
print best_rf
print r2_score(y_pred=y_pred, y_true=y_test)

# visulize the result
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[-1.0, 1.0])

# print cnf_matrix
classes = [-1.0, 1.0]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes, title='Confusion matrix, without normalization')
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