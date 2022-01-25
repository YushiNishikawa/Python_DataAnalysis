#%%
from operator import imod
from tkinter import E
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

boston = load_boston()

#%%
data_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
data_boston["PRICE"] = boston.target

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

X_train, X_test, y_train, y_test = train_test_split(
    data_boston_x, data_boston_y, test_size=0.3, random_state=123
)

lr_multi.fit(X_train, y_train)
y_pred = lr_multi.predict(X_test)
#%%
# MSE
mean_squared_error(y_pred, y_test)
# RMSE
np.sqrt(mean_squared_error(y_pred, y_test))

#%%
# MAE
mean_absolute_error(y_test, y_pred)


#%%
# MSLE
mean_squared_log_error(y_test, y_pred)
# RMSLE
np.sqrt(mean_squared_log_error(y_test, y_pred))

# %%
