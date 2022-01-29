#%%

import jupyter
import pandas as pd
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import tf_contextlib
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X_dftrain, y_dftrain = fetch_openml("mnist_784", version=1, return_X_y=True)
X_train = X_dftrain.values
y_train = y_dftrain.values

X_train = X_train.reshape(70000, 28, 28)

print(X_train.shape)
plt.imshow(X_train[0], cmap=plt.cm.gray_r)
plt.imshow(X_train[1], cmap=plt.cm.gray_r)

#%%
print(y_train.shape)
print(y_train[0])
print(y_train[1])

X_train = X_train.reshape(70000, 28 * 28)
X_train = X_train.astype("float32") / 255


X_train, X_test, y_train, y_test = train_test_split(
    X_train,
    y_train,
    stratify=y_train,
    random_state=0,
)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Dense(64, activation="relu", input_dim=28 * 28))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64)
model.evaluate(X_test, y_test)

#%%

model2 = Sequential()
model2.add(Dense(64, activation="relu", input_dim=28 * 28))
model2.add(Dense(10, activation="softmax"))

model2.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
model2.fit(X_train, y_train, epochs=5, batch_size=64)
model2.evaluate(X_test, y_test)

#%%
