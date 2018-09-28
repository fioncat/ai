# 公式的Latex源码

一些公式推导无法在GitHub显示,故将源码贴在这里.

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
