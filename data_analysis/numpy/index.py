import numpy as np

# Use slices to get subarrays.
# Changes to the subarrays will be reflected to the original array
arr = np.array([1, 2, 3, 4, 5, 6])
subarr = arr[0:2]
print("original array = ", arr)
print("subarray = ", subarr)
subarr[1] = 99
print("after chaned subarray, original array = ", arr)

# If do not want to use references to affect the original array,
# can use copy() method to copy the array.
arr = np.array([1, 2, 3, 4, 5, 6])
subarr = arr[1:3].copy()
subarr[1] = 99
print("after chaned copied subarray, original array = ", arr)

# For 2D array, a single index gets an array,
# two indexes get the value.
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("original 2D array:")
print(arr)
print("get 2nd line of the array:")
print(arr[1])
print("get (2, 0) element:")
print(arr[2][0])

# Slicing a multidimensional array
print("Get the first two rows of the array:")
print(arr[:2])
print("Get the first two rows and the first two columns of the array:")
print(arr[:2, :2])

print("Get the 2nd line and the first two columns of the array:")
print(arr[1, :2])

# A sperate colon indicates the entire lines or columns:
print("Get all lines and the first two columns of the array:")
print(arr[:, :2])
print("Get all columns and the last two lines of the array:")
print(arr[-2:, :])

# Modify the array by slicing.
arr[:2, :2] = 0
print("After changed array = ")
print(arr)


