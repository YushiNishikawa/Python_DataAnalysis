#%%

from pydoc import describe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
data_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
data_iris["target"] = iris.target

#%%
import seaborn as sns

sns.pairplot(data_iris, hue="target")

#%%
scaler = StandardScaler()
data_std = scaler.fit_transform(data_iris[iris.feature_names])
data_std_df = pd.DataFrame(data_std, columns=data_iris.columns[0:4])

# original data
print(data_iris.describe())
# Standardization
print(data_std_df.describe())

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(data_std)
print(pca_transformed.shape)

plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=data_iris["target"])
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

print(pca.components_)

data_pc = pd.DataFrame(
    pca.components_,
    columns=data_iris.columns[0:4],
    index=["PC{}".format(x + 1) for x in range(pca.n_components)],
)
print(data_pc)
# %%
