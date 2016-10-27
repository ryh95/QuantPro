from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from Lecture5.tools.plot_confusion import plot_confusion_matrix
import pandas as pd

def get_tuned_svc(X_train, y_train):
    # Set the parameters by cross-validation
    tuned_parameters = [
            {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
        ]

    # Perform the grid search on the tuned parameters
    model = GridSearchCV(SVC(C=1,random_state=2), tuned_parameters, cv=10)
    model.fit(X_train, y_train)

    # choose best model
    best_model = model.best_estimator_
    return best_model

def get_tuned_random_forest(X_train,y_train):
    # Set the parameters by cross-validation
    tuned_parameters = [
        {'n_estimators': [5,10,20,50,100], 'max_depth': [1,2,3]}
    ]

    # Perform the grid search on the tuned parameters
    model = GridSearchCV(RandomForestClassifier(n_jobs=4,random_state=2), tuned_parameters, cv=10)
    model.fit(X_train, y_train)

    # choose best model
    best_model = model.best_estimator_
    return best_model