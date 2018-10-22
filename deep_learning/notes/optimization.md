# 深度学习中的优化笔记

深度学习中的优化问题是非常重要的.深度学习动辄就需要用到多台机器训练数天甚至数月的时间,而且仅仅是训练单个神经网络.如何设计好的优化方法,让神经网络能够更快地,更准确地找到损失函数的全局最低点已经成为了一个很重要的问题.

## 神经网络的挑战

在传统的机器学习算法中,我们一般会小心地设计目标函数和约束,使其尽量是凸的.从而避免优化问题的复杂度.然而在神经网络中,非凸情况是不可避免的.

我们来看在训练神经网络的过程中可能出现的问题.

### 病态

病态问题在凸优化中非常常见.最突出的是Hessian矩阵的病态.(详见凸优化)

在神经网络中,病态体现在梯度下降会收敛于某些情况.在使用代价函数的泰勒展开预测其梯度的时候,可以得到:

$$f(x-\epsilon g)\approx f(x)-\epsilon g^tg+\frac{1}{2}g^THg$$

当$\frac{1}{2}g^THg$超过$\epsilon g^Tg$的时候,代价函数反而增加了.也就是说,在某些情况下,梯度的范数可能会随时间增加,而不是像我们期望的那样收敛到最小.这时就产生了病态问题.此时训练将没有效果.

### 局部极小值

不像凸函数,神经网络中的代价函数可能存在着多个局部极小值点.

然而神经网络的局部极小值点并不是非凸造成的.如果一个足够大的训练集能够唯一确定一个模型的参数,那么我们称这个模型是**可辨识**的.但是对于神经网络来说,我们可以交换单元的权重而不改变模型的输出,所以神经网络是**不可辨识**的模型.

不可辨识的典型特征就是神经网络具有非常多甚至无限多个局部极小值,因为隐藏层的局部极小点会被反应在总的代价函数上.这些局部极小值都有相同的代价函数值.

如果局部极小值比全局极小值有很大的代价,那么基于梯度的优化算法会带来极大的问题.但是对于现实问题中的局部极小值,很多学者认为大部分局部极小值还是有很小的代价函数,所以是否找到真正的全局最小值并不是这么重要.重要的是找到在参数空间中代价很小的值.

### 鞍点

对于高维的非凸随机函数,鞍点出现的实际上比局部极小值出现的概率要大.在鞍点处,Hessian矩阵同时具有正负特征值.处于正特征值对应的特征向量的方向的代价更高(代价函数的函数值更大),而负特征值对应的特征向量方向的代价更低.

鞍点的某个横截面是局部极大点,另外一个横截面是局部极小点.

Hessian矩阵在局部极小点处只有正特征值(正定的),在多维情况下,Hessian矩阵正定的概率会很小,而其既有正特征又有负特征的概率明显大于其正定的概率.所以说在高维情况下局部极小点比鞍点要少很多.

但是在代价较低的时候,Hessian是正定的概率又会高了很多.因为此时已经逼近临界点了.所以我们可以根据Hessian是否正定来判断此时代价是达到于局部极小点还是鞍点.

梯度下降在鞍点处的下降速度将会非常慢,甚至可能会卡在这里.后面介绍的SGD算法可以在一定程度上逃离鞍点.

### 梯度爆炸

高度非线性的深度网络(循环神经网络极为常见)的代价函数通常包含由多个参数连乘而导致的参数尖锐的非线性(悬崖).这些区域的导数会非常大,梯度下降将一次更新过多而导致之前做的所有优化都变成无用功.

循环神经网络的梯度爆炸问题尤为明显,所以需要引入长期记忆.

### 长期依赖

当计算图变得非常深的时候,模型可能会在持续的学习中丧失了学习到先前信息的能力,让优化变得非常困难.这个问题在循环神经网络中非常明显(循环神经网络的计算图非常深,并且共享参数).

我们从数学上感受一下,加入某个计算图中有一个反复和矩阵$W$相乘的路径,那么在$t$步之后,相当于乘了$W^t$.假设矩阵$W$的特征分解为$V\mathbf{diag}(\lambda)V^{-1}$.那么有:

$$W^t=(V\mathbf{diag}(\lambda)V^{-1})^t=V\mathbf{diag}(\lambda)^tV^{-1}$$

假如$t$非常大,则当特征值$\lambda_i$稍微比1大一些,就会出现梯度爆炸;特征值稍微比1小一些,就会出现梯度消失.

循环网络重复使用相同的参数多次,所以很容易产生长期依赖,而前馈网络因为不使用重复的参数,所以即使使用非常深的结构,也能很大程度避免长期依赖问题.

## 基本算法

对于最传统的梯度下降法,这里不再介绍,而是直接介绍基于梯度下降的一些改进算法.

### 随机梯度下降

随机梯度下降(SGD)和其变种是机器学习中用得最多的优化算法,特别是在深度学习中.SGD按照数据生成分布抽取$m$个小批量样本,计算它们的梯度均值,可以得到梯度的无偏估计.

SGD的优势在于它的速度比传统的梯度下降要快很多,并且可能可以让代价逃离诸如鞍点的比较尴尬的点(用样本去估计梯度使得梯度不再那么依赖于当前的代价点).

SGD算法的伪代码为:

***

$\mathbf{def\ \ }SGD(\epsilon\leftarrow\mathbf{learning\ \ rate},\theta\leftarrow\mathbf{initial\ \ parameter}):$

&ensp;&ensp;&ensp;&ensp;$\mathbf{while\ \ epoch\ \ dose\ \ not\ \ end}:$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\lbrace(x^{(1)},y^{(1)}),\dots,(x^{(m)},y^{(m)})\rbrace\leftarrow\mathbf{small\ \ batch\ \ from\ \ training\ \ data}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\hat{g}\leftarrow\frac{1}{m}\nabla_\theta\sum_i^mL(f(x^{(i)};\theta),y^{(i)})$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\theta\leftarrow\theta+\epsilon\hat{g}$

***

在传统的梯度下降中,学习率是固定的.但是对于SGD来说,有必要随着时间逐渐降低学习率.这是因为SGD中引入了噪声源($m$个从训练集中采集的样本)的梯度并不会在极小值处消失.而使用完整的训练集在接近极小值处梯度会变得非常小,之后变为0.因此不使用SGD可以有固定的学习率,而SGD在后期应该让学习率很低以保障收敛.

在实践中,一般会对学习率进行线性衰减直到第$\tau$次迭代:

$$\epsilon_k=(1-\alpha)\epsilon_0+\alpha\epsilon_\tau$$

在使用这个策略的时候,我们需要注意的参数是$\epsilon_0$,$\epsilon_\tau$和$\tau$.一般$\epsilon_\tau$大概设为$\epsilon_0$的1%.

对于$\epsilon_0$,如果太大,学习曲线会剧烈地振荡,代价函数可能会明显地增加.如果太小,学习速度会非常缓慢,学习可能会卡在一个相当高的代价值.

通常可以检测前几轮迭代,选择一个比在效果上表现最佳的学习率更大的学习率,但又不会导致严重的振荡.

### 动量

SGD在某些情况下可能会下降地很慢.动量方法可以加速我们的学习,特别是在遇到高曲率,小而一致的梯度,或是带噪声的梯度.

动量算法积累了之前梯度指数级衰减的移动平均,并沿着该方向移动.计算动量的公式是:

$$v\leftarrow\alpha v-\epsilon\nabla_\theta(\frac{1}{m}\sum^m_{i=1}L(f(x^{(i)};\theta),y^{(i)}))$$

动量由之前的梯度决定,如果$\epsilon$和$\alpha$的值越大,则之前梯度对当前动量的影响就越大.

动量可以帮助我们翻过代价的一些"山谷"(曲率很大但是很陡峭的地方),它往往能够直接越过这些山谷,而传统的SGD可能会困在山谷中移动很久,甚至可能会最终收敛于山谷.

采用动量的SGD参数的伪代码为:

***

$\mathbf{def\ \ mometumSGD(\epsilon,\ \theta,\ \alpha,\ }v\leftarrow\mathbf{initial\ \ mometum}):$

&ensp;&ensp;&ensp;&ensp;$\mathbf{while\ \ epoch\ \ dose\ \ not\ \ end:}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\lbrace(x^{(1)},y^{(1)}),\dots,(x^{(m)},y^{(m)})\rbrace\leftarrow\mathbf{small\ \ batch\ \ from\ \ training\ \ data}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\hat{g}\leftarrow\frac{1}{m}\nabla_\theta\sum_i^mL(f(x^{(i)};\theta),y^{(i)})$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$v\leftarrow\alpha v-\epsilon g$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\theta\leftarrow\theta+v$

***

### Nesterov动量

Nesterov动量是传统动量的一个变种,它计算动量的规则变为了下面这样:

$$v\leftarrow\alpha v-\epsilon\nabla_\theta(\frac{1}{m}\sum^m_{i=1}L(f(x^{(i)};\theta+\alpha v),y^{(i)}))$$

也就是说,在真正地更新参数之前,先在参数上施加之前的动量.伪代码如下:

***

$\mathbf{def\ \ NesterovMometumSGD(\epsilon,\ \theta,\ \alpha,\ }v\leftarrow\mathbf{initial\ \ mometum}):$

&ensp;&ensp;&ensp;&ensp;$\mathbf{while\ \ epoch\ \ dose\ \ not\ \ end:}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\lbrace(x^{(1)},y^{(1)}),\dots,(x^{(m)},y^{(m)})\rbrace\leftarrow\mathbf{small\ \ batch\ \ from\ \ training\ \ data}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\tilde{\theta}=\theta+\alpha v$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\hat{g}\leftarrow\frac{1}{m}\nabla_\theta\sum_i^mL(f(x^{(i)};\theta),y^{(i)})$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$v\leftarrow\alpha v-\epsilon g$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\theta\leftarrow\tilde{\theta}+v$

***

## 参数化初始策略

神经网络算法基本都是迭代的.这样一来参数的初始化就是一件很重要的事情.初始点的选择能决定算法是否收敛,如果初始点不稳定,算法可能会遭遇数值困难,最终无法收敛.

通常情况下,可以给参数中的偏置设置启发式挑选的常数,对权重进行随机初始化.我们通常使用高斯或均匀分布去初始化权重.高斯和均匀之间的差别通常不会很大.但是分布的大小对于优化和模型的泛化能力确实有很大的影响.

关于参数的选择,实际上优化和正则化是矛盾的.比较大的参数对于优化比较有利,这样优化可以学习到更多地梯度.而参数过大可能会导致模型的泛化误差较高,甚至可能对于输入过于敏感.因此初始化参数的范围设置通常不是什么简单的事情.

我们可以吧初始化参数的范围当做超参数去调整,使用诸如超参数搜索法选择一个合适的范围.

当然,有一些启发式方法能够帮助我们选择初始化参数,例如,对于一个$m$个输入和$n$个输出的全连接层,可以将权值初始化为:

$$W_{i,j}\sim U(-\sqrt{\frac{6}{m+n}},\sqrt{\frac{6}{m+n}})$$

当然,这个方法并不是万能的,也没有足够的理论依据证实它是有效的,仅仅可以作为我们的参考.

## 自适应学习算法

学习率在神经网络中通常是一个非常难以调整的超参数.它对模型的性能和最终的结果有着显著的影响.损失函数通常对某些方向过于敏感而导致代价的下降速度非常慢.

动量可以在一定程度上解决这个问题,但是它带来了另外一个超参数,如果调整这个超参数又是一个新的问题.所以我们渴望在不引入其他超参数的情况下让模型更快地收敛,同时又能逃离一些局部最低点.

自适应算法可以根据代价的情况自动调整学习率,让学习率能够适应当前的模型,使模型的收敛速度更快.

### AdaGrad

**AdaGrad**独立地适应所有模型参数的学习率,缩放每个参数反比于其所有梯度历史平方值总和的平方根.

AdaGrad的效果是,在参数空间中一些平缓的倾斜方向会取得更大的进步,也就是在平缓的地方代价可以下降地更快.AdaGrad的伪代码是:

***

$\mathbf{def\ \ AdaGrad(}\epsilon,\ \theta,\ \delta\leftarrow\ \mathbf{a\ small\ constant,\ usually\ set\ about}\ 10^{-7}):$

&ensp;&ensp;&ensp;&ensp;$\mathbf{while\ \ epoch\ \ dose\ \ not\ \ end:}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$r=0\leftarrow\mathbf{Gradient\ cumulative\ variable}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\mathbf{calculate\ }g$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$r\leftarrow r+g\ \odot\ g\leftarrow\mathbf{Cumlative\ squared\ gradient}$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\Delta\theta\leftarrow-\frac{\epsilon}{\delta+\sqrt{r}}\ \odot\ g$

&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;$\theta\leftarrow\theta+\Delta\theta$

***

然而,在训练开始时积累梯度平方会导致有效学习率过早和过量的减少.导致最终无法收敛于代价比较低的点.

### RMSProp

**RMSProp**修改AdaGrad使其在非凸情况下效果更加好.RMSProp使用指数衰减平均来丢弃那些遥远的历史.使其在找到碗状结构后快速收敛,伪代码如下:

***

$\mathbf{def\ \ RMSProp(\epsilon,\ \theta,\ \delta,\ }p\leftarrow \mathbf{Decay\ rate}):$

***

### Adam

## 二阶近似方法

### 牛顿法

### 共轭梯度

### BFGS
