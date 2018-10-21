# TensorFlow 入门笔记

TensorFlow是Google推出的著名开源计算框架.被大量应用于机器学习和深度学习.TensorFlow支持多平台(Linux, MacOS, Windows, Android, IOS).支持GPU加速,并行计算等等.目前已经被大量应用于深度学习领域.

注意,本文仅包括TensorFlow深度学习的笔记,对于一些其他工具,建议自行查阅文档.

本笔记仅仅是TensorFlow的基本思想,充其量只是TensorFlow核心的冰山一角,更多TensorFlow内容请参见TensorFlow[官网](https://www.tensorflow.org/)和[TensorFlow Python API文档](https://www.tensorflow.org/api_docs/python/).

## 计算图

TensorFlow是通过计算图的形式进行计算的.Tensor表示**张量**,它是TensorFlow中最基础的单位,是计算图中的节点.在程序中,它可以是标量,向量或是矩阵.Flow表示了张量在计算图中通过计算互相转换,好像Tensor在图中流动一样.

tensor之间的依赖关系就是通过计算图来描述的,而我们在使用TensorFlow编程的时候最重要的步骤就是定义计算图.

TensorFlow程序默认维护了一个计算图,默认情况下我们创建的tensor都是在这个计算图中的:

```python
import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([3.0, 4.0], name='b')

result = tf.add(a, b)
```

这就在默认的计算图中创建了3个tensor,前两个tensor是常量,result的结果依赖于a和b的和.

通过tensor的graph属性,我们可以获取这个tensor的计算图对象.TensorFlow可以通过`get_default_graph()`来获得默认的图环境,因此下面输出为`True`:

```python
print(a.graph == tf.get_default_graph())
```

我们可以通过`tf.Graph()`手动创建不同的图,一定要注意,tensor在不同的图之间是不共享的:

```python
g1 = tf.Graph()
with g1.as_default():
    ...

g2 = tf.Graph()
with g2.as_default():
    ...
```

如果我们在`g1`和`g2`中创建了两个tensor,即使它们的名字是一样的,但是因为在不同的图中,所以它们可以拥有自己不同的值和或不同的依赖.

我们甚至可以通过`tf.Graph.device`来制定运行这个图的设备:

```python
g = tf.Graph()
with g.device('/gpu:0'):
    ...
```

以上的图在运行的时候就会跑在GPU上了(前提是安装好tensorflow-gpu).

## 张量

Tensor是TensorFlow最基本的单位.所有的数据在TensorFlow中都是以Tensor的形式表示的.

我们在定义Tensor的时候,并不会直接就计算出Tensor的值,而是仅仅保存它的结构,即这个Tensor是怎么计算出来的,在上一节的第一个例子中,`result`是`a`和`b`的和,而定义完`result`之后它的结果并没有被计算出来,此时`result`仅仅保存了它的结果是依据`a`和`b`计算而来的.这和传统的python编程是完全不一样的.

Tensor中还保存了一些重要的属性:

- name: Tensor在图中的唯一标识,它由两个部分组成,格式是"node:src_output".其中node是由用户在定义Tensor的时候通过name属性指定的,不指定的话使用默认值;src_output表示这个张量来自节点的第几个输出.
- type: Tensor储存的数据类型.在默认情况下,整数会使用`tf.int32`,小数使用`tf.float32`.我们也可以通过`dtype`属性手动指定.注意,对类型不匹配的Tensor执行一些运算操作TensorFlow会报错.
- shape: Tensor储存的数据的维度.格式和NumPy一样.对于标量,shape是一个空的元组.通过Tensor的`get_shape()`可以获取维度.

Tensor被分为了多个类型,上面我们已经见识过`tf.constant()`了,它是一个常量,需要在定义的时候就初始化好.

另外一个类型是`tf.placeholder()`.这也是一个常量,不过它不用事先初始化,仅仅作为一个占位符存在.我们只需要指定它的类型,维度即可.在稍后真正执行图(后面会介绍怎么执行计算图)的时候,才对`placeholder`进行赋值:

```python
import tensorflow as tf

a = tf.placeholder(shape=(None, 12), dtype=tf.float32)
a = tf.placeholder(shape=(None), dtype=tf.int32)
```

这里注意在维度中传入`None`,这表示此维度可以是任意值.例如对于`a`,维度为`(128,12)`和`(200,12)`的数据都是合法的.而对于`b`,只要是一维的数据,都是合法的.

另外一个Tensor就是变量了.它是根据其他Tensor计算而来的.变量可以通过`tf.Variable()`创建,我们后面会详细介绍变量.

## 会话

Tensor在定义的时候并不会计算值,而如果我们想要计算一个Tensor的结果,就需要使用到Session了.Session会真正地去执行一个计算图并计算其中Tensor的值.

调用`tf.Session()`可以创建一个会话,会话一般需要关闭,所以我们一般使用`with`去创建.调用会话的`run()`可以执行计算图,计算其中一个Tensor的具体值.

以下的代码演示了如何创建一个会话并计算加法:

```python
import tensorflow as tf

a = tf.constant(1.0)
b = tf.constant(2.0)

c = tf.add(a, b)

with tf.Session() as sess:
    print(tf.run(c))
```

上面就真正地计算了`c`的值并且返回结果,程序理所当然地输出了`3.0`.

如果计算图中存在`tf.placeholder()`,那么在`run()`的时候需要给它们赋值(否则会报错),赋值通过一个字典`feed_dict`完成:

```python
import tensorflow as tf

a = tf.placeholder(dtype=tf.int32)
b = tf.placeholder(dtype=tf.int32)

c = tf.add(a, b)

with tf.Session() as sess:
    print(sess.run(c, feed_dict={a: 10, b:20}))
```

因为`c`的计算必须依赖`a`和`b`的值,所以在执行计算图之前必须通过`feed_dict`来给`a`和`b`赋值.这样,程序就输出了`30`.

## 变量

神经网络中包含了很多参数,包括权值和偏差.这些参数在神经网络的训练过程中是要不停地改变的.所以,TensorFlow使用变量去组织,保存这些参数.

变量需要赋予一个初始值.在定义神经网络的时候,我们一般使用随机值去初始化它,TensorFlow提供了以下常见的随机数生成工具:

- `tf.random_normal()`: 生成一组正态分布的随机数,常见参数为:
  - shape: 随机数维度.
  - mean: 正态分布的均值,默认为`0.0`.
  - stddev: 正态分布标准差,默认为`1.0`.
- `tf.truncated_normal()`: 生成正态分布随机数,但是如果随机数的值偏离均值超过2个标准差,随机将会重新进行.它的参数和`tf.random_normal()`一样.
- `tf.random_uniform()`: 生成均匀分布.
- `tf.random_gamma()`: 生成Gamma分布.

我们也可以直接使用常数去初始化变量,常见的工具有:

- `tf.zeros()`: 产生一个全0的数组.
- `tf.ones()`: 产生一个全1的数组.
- `tf.fill()`: 产生一个全部是给定数字的数组.
- `tf.constant()`: 产生一个常量.

关于更多参数和具体用法,请参考TensorFlow文档.

在神经网络中,`weights`通常用正态分布去初始化,`biases`通常用全0值去初始化:

```python
weights = tf.Variable(tf.random_normal([2, 3], stddev=2))
biases = tf.Variable(tf.zeros[3])
```

上面的代码中,`weights`被初始化为了一个2x3的矩阵,其中的值满足均值为0,标准差为2的正态分布,`biases`被初始化为了拥有3个0的向量.

我们也可以使用其它变量的值去初始化另一个一个变量:

```python
a1 = tf.Variable(a.initialized_value())
a2 = tf.Variable(a.initialized_value() * 2)
```

这样的话`a1`的初始值和`a`一样,`a2`的初始值是`a`的两倍.

在变量之前,我们需要先让Session把变量的初始化过程跑完.这可以`sess.run(v.initializer)`完成.但是如果变量很多的话一个一个初始化就很麻烦.所以可以直接通过`sess.global_variables_initializer()`完成.这是一个操作,把它传给`run()`可以一次性初始化计算图中的所有变量.

我们之前说过,变量也是一种Tensor,不过它的值是通过其它Tensor计算而来的.更加通用的,我们可以把这里的"计算"抽象成"操作".而在TensorFlow内部,变量必须是由操作生成的.所以实际上上面的初始化也是操作的过程(这个操作的输入不是Tensor,而是随机函数,我们把这个操作叫做Assign),而任何操作都需要`sess.run()`才能执行,这就解释了为什么变量的初始化需要单独进行`run()`了.

Tensor之间可以通过各种操作得到变量,例如`tf.add()`对两个Tensor求和得到一个变量,`tf.matmul()`对两个矩阵相乘.这样的数学计算操作非常多,详情可以查看文档.

类型对于变量来说是非常重要的,在赋值或者操作的时候一定要注意类型是否合法.例如,不能把一个`float32`的变量赋值给一个`int32`的变量,如果这么做,TensorFlow会报错.

## 实现线性模型

线性模型是

## 实现神经网络
