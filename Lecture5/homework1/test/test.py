import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Lecture5.homework1.model import get_tuned_random_forest_classifier
from Lecture5.tools.plot_confusion import plot_confusion_matrix
import numpy as np

df = pd.read_csv('hs300_dataset.csv')

df.index = pd.to_datetime(df['date'])
df.drop('date',axis=1,inplace=True)

X = df.iloc[:,[20,12,6,11,21,7,5,22,26,14,2,27,16,24,18,4,0,1]]
y = df['direction']
# split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

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

