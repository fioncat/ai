import pandas as pd
import numpy as np

# DataFrame相当于矩阵的数据表,包含行索引和列索引
# 可以通过保存等长列表的字典来创建DataFrame
# 每个key-value对是一列
data = {
    'year': [2018, 2019, 2020, 2021, 2022],
    'cost': [20000, 70000, 20000, 30000, 10000],
    'boss': ['John', 'Jack', 'Mike', 'John', 'John']
}
df = pd.DataFrame(data=data)
print("create dataframe by dictonary:")
print(df)

# 在这种情况下,行索引是自动分配的,并且列会排好序
# head方法会只显示前5行数据
data2 = {
    'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'b': [6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
}
df = pd.DataFrame(data=data2)
print("Use head() to show top five lines data:")
print(df.head())

# 如果指定列顺序, DataFrame会按照指定顺序排好序
df = pd.DataFrame(data=data, columns=['boss', 'cost', 'year'])
print("Assign the sequence of the index:")
print(df)

# 通过index,可以指定行索引
df = pd.DataFrame(data=data, columns=['boss', 'cost', 'year', 'what'],
                  index=['a', 'b', 'c', 'd', 'e'])
print("Assign line index:")
print(df)
# 可见,对于没有出现的列,所有值都会设置为NaN

# 通过索引可以取出一列数据(实际上返回的就是一个Series)
print("Get one column data:")
print(df['boss'])

# 通过两个索引可以取出一个元素
print("Get one element by two index:", df['boss']['a'])

# 也可以以属性的方式直接获取Series
print("Get one column data by attribute:")
print(df.year)
