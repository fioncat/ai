import numpy as np

# Transpose operation
arr = np.arange(15).reshape((3, 5))
print("original array = ")
print(arr)

print("Transpose:")
print(arr.T)

# Generic function, performs an action on each element of an array.
# Forexample, sqrt and exp function:
arr = np.arange(8).reshape((2, 4))
print("original array = ")
print(arr)

print("Use sqrt = ")
print(np.sqrt(arr))

print("Use exp = ")
print(np.exp(arr))

# Use where can achieve Ternary Expression:
cond = np.array([True, False, True, True, False])
arr1 = np.array([1, 1, 1, 1, 1])
arr2 = np.array([0, 0, 0, 0, 0])
print("Pick arr1 if True, arr2 if False = ", np.where(cond, arr1, arr2))

# where also can use to replace values when they fit given condition.
arr = np.random.randn(4, 4)
print("random array = ")
print(arr)
print("If value > 0, replace to 1, else replace to 0 = ")
print(np.where(arr > 0, 1, 0))

# Some statistical functions:
arr = np.random.randn(5, 6)
print("ramdom array = ")
print(arr)
print("mean = ", np.mean(arr))
print("sum = ", np.sum(arr))

# Use param axis can generate column or row statistics
print("mean, axis=1 = ", np.mean(arr, axis=1))
print("mean, axis=0 = ", np.mean(arr, axis=0))

# True will be treat as 1, and False is 0.
# So can use sum to calculate the number of True:
arr = np.array([1, -1, -9, 23, -7, -10])
print("arr = ", arr)
print("The number of positive numbers in arr = ", np.sum(arr > 0))

# any() return True If there is at least one is True in Boolean Array.
# all() return True If all is True in Boolean Array.
print("Use any to arr > 0 = ", np.any(arr > 0))
print("Use all to arr > 0 = ", np.all(arr > 0))

# Sort single-dimensional array:
arr = np.random.randn(6)
print("original array = ", arr)
print("sorted array = ", np.sort(arr))

# Sort multiple-dimensional array:(by row or column)
arr = np.random.randn(4, 5)
print("original array = ")
print(arr)
print("sort by row = ")
print(np.sort(arr, axis=0))
print("sort by column = ")
print(np.sort(arr, axis=1))

