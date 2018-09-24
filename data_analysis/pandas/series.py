import pandas as pd
import numpy as np

# Series储存序列,但是每个元素都有一个索引
# 在不指定索引的情况下,索引是0到N-1
data = pd.Series([4, -2, 10, 29])
print("create a default index of Series:")
print(data)

# 通过index可以指定每个数据的索引
data = pd.Series([4, -2, 10, 29], index=['a', 'b', 'c', 'd'])
print("Specify index:")
print(data)

# 可以直接通过索引取出数据,像dictionary那样
print("pick index='a':", data['a'])

# 索引可以是数组,这样会取出所有对应的值,返回一个新的Series
print("Use array as index:")
print(data[['a', 'c', 'd']])

# 在索引中可以进行比较
print("Use comparison statement(>=10):")
print(data[data >= 10])

# 可以对Series直接进行一些科学计算(可以使用numpy的函数)
print("data * 2:")
print(data * 2)

print("exp:")
print(np.exp(data))

# Series可以看成一个长度固定并且有序的字典
# 可以使用一些字典用到的操作符
print("use 'in':")
print('a' in data)
print('z' in data)

# Series和字典如此像,我们可以使用字典来生成一个Series
sdata = {'a': 10, 'b': -20, 'c': 300, 'd': -12}
data = pd.Series(sdata)
print("Use dictionary to generate series:")
print(data)

# 我们可以手动指定字典中元素的位置,通过传入index
data = pd.Series(sdata, index=['a', 'c', 'd', 'b', 'f'])
print("Pass index when using dictionary to create series('f' is not in the dictionary):")
print(data)

# 上面的'f'并没有出现在sdata中,因此Series在f的位置保存了NaN.我们可以通过isnull和notnull来判断
# 一个series中是否存在NaN数据
print("use isnull:")
print(data.isnull())

print("use notnull:")
print(data.notnull())

# 我们可以将多个series进行计算:
data1 = pd.Series({'a': 10, 'b': 20, 'c': 30})
data2 = pd.Series({'a': -1, 'b': -2, 'c': -8, 'd': 80})
print("data1 * data2 = ")
print(data1 * data2)
# 对于在一个Series中出现缺没有在另一个中出现的元素,Pandas会赋值为NaN

# Series的索引可以按照位置赋值来改变
data = pd.Series([100, 200, 300, 400])
data.index = ['a', 'b', 'c', 'd']
print("Change index:")
print(data)
