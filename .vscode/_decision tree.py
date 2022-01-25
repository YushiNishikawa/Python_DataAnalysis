#%%

from ast import increment_lineno
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

clf = DecisionTreeClassifier(max_depth=3, criterion="gini")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

from dtreeviz.trees import dtreeviz

viz = dtreeviz(
    clf,
    X,
    y,
    feature_names=iris.feature_names,
    target_name="breed",
    class_names=[str(i) for i in iris.target_names],
)

viz.view()
