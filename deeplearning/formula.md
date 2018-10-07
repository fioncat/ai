# 公式的Latex源码

一些公式推导无法在GitHub显示,故将源码贴在这里.

## 感知器

$$
\begin{aligned}
    \frac{\partial}{\partial{w_j}}\hat{y}&=\frac{\partial}{\partial{w_j}}\sigma(Wx+b)\\
    &=\sigma(Wx+b)(1-\sigma(Wx+b))\cdot\frac{\partial}{\partial{w_j}}(Wx+b)\\
    &=\hat{y}(1-\hat{y}) \cdot x_i
\end{aligned}
$$

$$

\begin{aligned}
    \frac{\partial}{\partial{w_j}}E&=\frac{\partial}{\partial{w_j}}(-\frac{1}{m}\sum^m_{i=1}y_ilog(\hat{y}_i)+(1-y_i)log(1-\hat{y}_i))\\
    &=-\frac{1}{m}\sum^m_{i=1}y_i\frac{\partial}{\partial{w_j}}log(\hat{y}_i)+(1-y_i)\frac{\partial}{\partial{w_j}}log(1-\hat{y}_i)\\
    &=-\frac{1}{m}\sum^m_{i=1}y_i(1-\hat{y}_i)x^{(i)}_j-(1-y_i)\hat{y}_ix^{(i)}_j\\
    &=-\frac{1}{m}\sum^m_{i=1}(y_i-\hat{y}_i)x^{(i)}_j
\end{aligned}

$$

## 神经网络

$$
W^{(1)}=

\begin{pmatrix}
    W_{11}^{(1)} & W_{12}^{(1)}\\
    W_{21}^{(1)} & W_{22}^{(1)}\\
    W_{31}^{(1)} & W_{32}^{(1)}
\end{pmatrix}

W^{(2)}=

\begin{pmatrix}
    W_{11}^{(2)}\\
    W_{21}^{(2)}\\
    W_{31}^{(2)}
\end{pmatrix}

X=

\begin{pmatrix}
    x_1\\
    x_2\\
    1
\end{pmatrix}

$$

$$

\nabla E=(\frac{\partial E}{\partial W^{(1)}},\frac{\partial E}{\partial W^{(2)}},\dots,\frac{\partial E}{\partial W^{(n)}})

$$

$$h_1=W^{(1)}_{11}x_1+W^{(1)}_{21}x_2+W^{(1)}_{31}$$

$$h_1=W^{(1)}_{12}x_1+W^{(1)}_{22}x_2+W^{(1)}_{32}$$

$$h=W^{(2)}_{11}\sigma(h_1)+W^{(2)}_{21}\sigma(h_2)+W^{(2)}_{31}$$

$$\hat{y}=\sigma(h)$$

aa

$$
\overline{s}_t=\phi(\overline{x}_t\cdot W_x+\overline{s}_{t-1}\cdot W_s)
$$

最终输出

$$
\overline{y}=\overline{s}_t\cdot W_y
$$
