import pandas as pd
import numpy as np

a = pd.Series([2, 3, 4, 5])
print(a)
print(a[0])
print(a[1:3])
print(a.sum())
print(a.mean())

col_1 = np.array(["A", "B", "A", "C", "D"])
col_2 = np.array(["1", "2", "3", "4", "5"])
col_3 = np.array(["3", "6", "7", "8", "2"])

df = pd.DataFrame({"col_1": col_1, "col_2": col_2, "col_3": col_3})
print(df.head())
print(df.shape)

print(df["col_1"])
print(df.iloc[2])
print(df.iloc[2:])
print(df.iloc[2, 1])
print(df[df["col_1"] == "A"])
print(df.query('col_1 == "B"'))

print(df.describe())
print(df.info())
print("sum = " + df["col_2"].sum())
print("average = " + str(df["col_2"].mean()))
print(df.sort_values("col_2", ascending=False))
df = df.replace("A", np.nan)
print(df)

col_1 = np.array(["A", "B", "A", "C", "D"])
col_2 = np.array([1, 2, 3, 4, 5])
col_3 = np.array([3, 6, 7, 8, 2])

print(df["col_2"].sum())
print(df["col_2"].mean())
print(df.sort_values("col_2", ascending=False))
df = df.replace("A", np.nan)
print(df)
print(df.dropna())
print(df.fillna(0))
