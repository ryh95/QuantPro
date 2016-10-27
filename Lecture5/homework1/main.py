import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from Lecture5.tools.indicators import EWMA ,ROC, CCI,ForceIndex
from Lecture5.tools.plot_confusion import plot_confusion_matrix

df = pd.read_csv('hs300.csv')

# # create features
def create_features(df,lags=5):
    X = df.set_index(df['date'])
    X.sort_values(by='date',inplace=True)
    X.drop('date',axis=1,inplace=True)
    X.index = pd.to_datetime(X.index)

    # add lags
    for i in xrange(0, lags):
        X["lag%s" % str(i+1)] = X["close"].shift(i+1)

    # add return
    X['return'] = X['close'].pct_change()*100.0
    for i in xrange(0,lags):
        X['return'+str(i+1)] = X['lag'+str(i+1)].pct_change()*100.0

    # add indicators
    X = EWMA(X,lags,close='close')
    X = ROC(X,lags,close='close')
    X = CCI(X,lags,high='high',low='low',close='close')
    X = ForceIndex(X,lags,close='close',volume='volume')

    # get rid of rows which contains nan values
    X.dropna(inplace=True)

    return X

df = create_features(df)

# feature rescale
df_norm = (df - df.mean()) / (df.max() - df.min())

# add direction as target value
df_norm['direction'] = np.sign(df['return'])

X = df_norm.iloc[:,:-1]
y = df_norm['direction']
# split data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

# Set the parameters by cross-validation
tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
    ]

# Perform the grid search on the tuned parameters
model = GridSearchCV(SVC(C=1), tuned_parameters, cv=10)
model.fit(X_train, y_train)

# choose best model
best_model = model.best_estimator_
y_pred = best_model.predict(X_test)
print best_model
print r2_score(y_pred=y_pred,y_true=y_test)

cnf_matrix = confusion_matrix(y_test,y_pred,labels=[-1.0,1.0])
# print cnf_matrix
classes = [-1.0,1.0]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes, title='Confusion matrix, without normalization')
plt.show()