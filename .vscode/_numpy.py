import numpy as np

arr = np.array([1, 2, 3, 4])
print(arr)
print(type(arr))
arr2 = np.array([[1, 2, 3, 4], [3, 6, 7, 8]])
print(arr2)
print(arr2.shape)
print(arr2[0])
print(arr2[1])
print(arr2[1][2])
print(arr2[1][1:3])

arr = np.array([1, 2, 3, 4])
print(arr + arr)
print(arr * arr)
print(5 + arr)
print(arr.sum())
print(arr.mean())
print(arr.max())

arr = np.arange(10)
print(arr)

arr.reshape(2, 5)
print(arr)
print(arr.reshape(2, 5))
arr = arr.reshape(2, 5)
print(arr)
