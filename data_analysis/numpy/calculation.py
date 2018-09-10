import numpy as np

# Calculation between ndarrays.
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("arr = ")
print(arr)
print("arr * arr = ")
print(arr * arr)
print("arr - arr = ")
print(arr - arr)

# Ndarray and scalar calculations, scala will be broadcast to all elements.
print("1 / arr = ")
print(1 / arr)
print("10 * arr = ")
print(10 * arr)
print("arr ** 2 = ")
print(arr ** 2)

# Compare two ndarrays, will return a boolean ndarray
arr2 = np.full((2, 3), 4)
print("arr2 = ")
print(arr2)
print("arr2 > arr = ")
print(arr2 > arr)
