#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
data_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
data_iris["target"] = iris.target

scaler = StandardScaler()
data_std = scaler.fit_transform(data_iris[iris.feature_names])
plt.scatter(data_std[:, 0], data_std[:, 1], c=data_iris["target"])

#%%
plt.scatter(data_std[:, 0], data_std[:, 2], c=data_iris["target"])

#%%
plt.scatter(data_std[:, 0], data_std[:, 3], c=data_iris["target"])

# %%
K_means = KMeans(n_clusters=2)
K_means.fit(data_std[:, [0, 1]])
print(K_means.labels_)
plt.scatter(data_std[:, 0], data_std[:, 1], c=K_means.labels_)
print(data_iris[K_means.labels_ == 0]["target"].value_counts())
print(data_iris[K_means.labels_ == 1]["target"].value_counts())


# %%
