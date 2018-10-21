# TensorFlow 入门笔记

TensorFlow是Google推出的著名开源计算框架.被大量应用于机器学习和深度学习.TensorFlow支持多平台(Linux, MacOS, Windows, Android, IOS).支持GPU加速,并行计算等等.目前已经被大量应用于深度学习领域.

注意,本文仅包括TensorFlow深度学习的笔记,对于一些其他工具,建议自行查阅文档.

## 计算图

TensorFlow是通过计算图的形式进行计算的.Tensor表示**张量**,它是TensorFlow中最基础的单位,是计算图中的节点.在程序中,它可以是标量,向量或是矩阵.Flow表示了张量在计算图中通过计算互相转换,好像Tensor在图中流动一样.

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

如果我们在`g1`和`g2`中创建了两个tensor,即使它们的名字是一样的,但是因为在不同的图中,所以它们可以拥有自己不同的值和或不同的依赖.

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

另外一个Tensor就是变量了.它可以根据其他Tensor计算而来,也可以是随机初始化的.变量需要通过`tf.Variable()`创建,我们后面会详细介绍变量.

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
