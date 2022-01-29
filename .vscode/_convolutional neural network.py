#%%

import pandas as pd
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.models import load_model, save_model
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

X_dftrain, y_dftrain = fetch_openml("mnist_784", version=1, return_X_y=True)
X_train = X_dftrain.values
y_train = y_dftrain.values

#%%

X_train = X_train.reshape(70000, 28, 28, 1)
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
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPool2D(2, 2))
model.summary()

#%%

model.add(Flatten())
model.add(Dense(32, activation="relu", input_dim=28 * 28))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=5, batch_size=64)
model.evaluate(X_test, y_test)

#%%
model.save_model("model.h5")
