#%%

from ast import increment_lineno
from tkinter.tix import Meter
from types import MethodDescriptorType
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

iris = load_iris()
tmp_data = pd.DataFrame(iris.data, columns=iris.feature_names)
tmp_data["target"] = iris.target
data_iris = tmp_data[tmp_data["target"] <= 1]
x_column_list = ["sepal length (cm)"]
y_column_list = ["target"]

X_train, X_test, y_train, y_test = train_test_split(
    data_iris[x_column_list],
    data_iris[y_column_list],
    test_size=0.2,
    random_state=123,
)

logit = LogisticRegression()
logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test,y_pred))
