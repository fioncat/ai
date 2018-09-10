import numpy as np

# generate random matrix.
data = np.random.randn(2, 3)
print("generate random 2x3 matrix: ")
print(data)

# check ndarray's shape.
print("use shape property: ", data.shape)
print("how many lines: ", data.shape[0])
print("how many columns: ", data.shape[1])

# check elements' type.
print("type: " , data.dtype)

# generate matrix:
data2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print("create a 2x4 matrix:")
print(data2)

# zeros and ones:
print("use zeros:")
print(np.zeros(10))
print(np.zeros((2, 4)))

print("use ones:")
print(np.ones(10))
print(np.ones((2, 4)))

# Use arange to create a sequence(Like range in Python lib.)
print("use arange:")
print(np.arange(15))

# other functions:
print("use eye:")
print(np.eye(5))

print("use full:")
print(np.full((3, 5), 66))

# specified type when create ndarray
arr3 = np.array([1, 2, 3], dtype=np.int32)
print("use dtype to specify element type:", arr3.dtype)

# convert data type.
arr4 = arr3.astype(np.float64)
print("After using astype to convert type from int32 to float64:")
print(arr4)

arr4 = np.array([1.2, 3.6, 7.8, 3.4], dtype=np.float64)
print("Convert from float to int, number after the decimal point will loss.")
print("origin:", arr4)
print("converted:", arr4.astype(np.int32))

# use astype convert string to number type.
arr = np.array(['3', '23', '100.5'], dtype=np.string_)
num_arr = arr.astype(np.float64)
print("convert from string to float:")
print(num_arr)

# NOTES: If the string can not convert to int, ValueError will be raised.

# You can pass type from a array to another.
arr1 = np.array([.1, .2], dtype=np.float)
arr2 = np.array([23, 12, 34], dtype=np.int)
print("pass type:")
print(arr2.astype(arr1.dtype))