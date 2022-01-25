#%%
from re import X
from statistics import linear_regression
import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.DESCR)

#%%
data_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
data_boston["PRICE"] = boston.target
print(data_boston.head())
print(data_boston.tail())


#%%
sns.jointplot("RM", "PRICE", data=data_boston)
sns.pairplot(data_boston, vars=["PRICE", "RM", "DIS"])

#%%
lr = LinearRegression()
x_column_list = ["RM"]
y_column_list = ["PRICE"]

data_boston_x = data_boston[x_column_list]
data_boston_y = data_boston[y_column_list]

lr.fit(data_boston_x, data_boston_y)
print(lr.coef_)
print(lr.intercept_)

# %%
lr_multi = LinearRegression()
x_column_list_for_multi = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]
y_column_list_for_multi = ["PRICE"]

data_boston_x = data_boston[x_column_list_for_multi]
data_boston_y = data_boston[y_column_list_for_multi]

lr_multi.fit(data_boston_x, data_boston_y)
print(lr_multi.coef_)
print(lr_multi.intercept_)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data_boston_x, data_boston_y, test_size=0.3, random_state=123
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lr_multi2 = LinearRegression()
lr_multi2.fit(X_train, y_train)
print(lr_multi2.coef_)
print(lr_multi2.intercept_)

y_pred = lr_multi2.predict(X_test)
print(y_pred)
print(y_pred - y_test)

# %%

from sklearn.metrics import mean_absolute_error

x_column_list = ["RM"]
y_column_list = ["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(
    data_boston[x_column_list],
    data_boston[y_column_list],
    test_size=0.3,
    random_state=123,
)
lr_single = LinearRegression()
lr_single.fit(X_train, y_train)
y_pred = lr_single.predict(X_test)

print(mean_absolute_error(y_pred, y_test))

x_column_list_for_multi = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
]
y_column_list_for_multi = ["PRICE"]

X_train, X_test, y_train, y_test = train_test_split(
    data_boston[x_column_list_for_multi],
    data_boston[y_column_list_for_multi],
    test_size=0.3,
    random_state=123,
)
lr_multi2 = LinearRegression()
lr_multi2.fit(X_train, y_train)
y_pred = lr_multi2.predict(X_test)

print(mean_absolute_error(y_pred, y_test))


#%%

from sklearn.linear_model import LinearRegression, Ridge, Lasso

lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)
lr_multi.fit(data_boston[x_column_list], data_boston[y_column_list_for_multi])

print(lr_multi.coef_)
print(lr_multi.intercept_)
X_train, X_test, y_train, y_test = train_test_split(
    data_boston[x_column_list_for_multi],
    data_boston[y_column_list_for_multi],
    test_size=0.3,
    random_state=123,
)
lr_multi2 = LinearRegression()
lr_multi2.fit(X_train, y_train)
y_pred = lr_multi2.predict(X_test)

# MAE
print(mean_absolute_error(y_pred, y_test))

lasso = Lasso(alpha=0.001, normalize=True)
lasso.fit(X_train, y_train)
print(lasso.coef_)
print(lasso.intercept_)

# Residual Error
y_pred_lasso = lasso.predict(X_test)

# MAE
print(mean_absolute_error(y_pred_lasso, y_test))

ridge = Ridge(alpha=0.001, normalize=True)
ridge.fit(X_train, y_train)
print(ridge.coef_)
print(ridge.intercept_)

# Residual Error
y_pred_ridge = ridge.predict(X_test)

# MAE
print(mean_absolute_error(y_pred_ridge, y_test))

