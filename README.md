# AI学习资料整理

对于AI的学习,我个人认为一定要理论结合实践,首先让自己的理论基础过关了,然后再去做一些小项目练手.

这个仓库就是本人在日常学习中以这种思路整理的笔记和一些小项目.

理论部分以Markdown文档为主,我在文档中使用了大量MathJax公式,而Github默认是不解析这些公式的,建议使用Chrome的插件[Github with MathJax](https://chrome.google.com/webstore/detail/github-with-mathjax/ioemnmodlmafdkllaclgeombjnmnbima/related).或者使用拥有MathJax引擎的Markdown阅读器阅读.

因为MathJax的粗体太难看了(不支持Latex中的bm),所以我在文中的向量并没有用传统的粗体风格.不过除了特殊说明,基本上所有的小写字母都是向量.对于矩阵,我还是使用加粗的方式表示.

实践是利用Python将算法转化为现实的可执行程序.Anaconda包含了本项目用到的绝大多数库,一些Anaconda没有包含的库如下:

- Tensorflow: Google的开源DeepLearning库, [官方网站](https://www.tensorflow.org/)
- Keras: 底层基于Tensorflow, 提供更加人性化的API. [文档传送门](https://keras.io/)

所有小项目基于Python 3.x编写.部分源代码使用jupyter notebook的编写.

## 数学

机器学习的理论要求一定数学基础.其中线性代数尤为重要,其次是概率论和数值优化的一些理论.所以我一开始整理的是一些数学相关的markdown笔记:

- [线性代数](mathematics/linear_algebra.md)
- [概率论](mathematics/probability_theory.md)
- [数值优化](mathematics/numerical_optimization.md)
- [凸优化]()

## 深度学习

深度学习属于机器学习的分支,它通过嵌套深层简单概念构造复杂的模型,以解决现实世界一些比较复杂的智能任务.深度学习要求有高等数学(简单的求导,偏导数,梯度等),线性代数和数值优化基础.

下面是深度学习的理论笔记(主要内容来自花书):

- [深度前馈网络Deep Feedforward Network](deep_learning/notes/mlp.md)
- [正则化]()
- [优化方法]()
- [卷积网络]()
- [循环网络]()
- [超参数]()

深度学习包含了以下的示例项目(建议使用GPU训练):

- [Python实现简单MLP模型并训练MNIST数据集]()
- [Tensorflow 实现MLP并训练MNIST数据集]()
- [Keras CNN训练CIFAR10数据]()
- [Tensorflow VGG迁移学习实现对花朵图片的分类]()
- [Keras VGG迁移学习实现对狗狗品种的预测]()
- [Keras VGG迁移学习实现Kaggle猫狗大战任务]()
- [Tensorflow RNN实现自动生成文章]()
- [Tensorflow RNN实现情感预测]()

## 机器学习

## 计算机视觉

## 自然语言处理
