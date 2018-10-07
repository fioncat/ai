# 梯度下降法公式

权值更新:

$$W=W+\alpha (-\frac{\partial E}{\partial W})$$

更改的权重

$$\Delta W_{ij}^k=-\alpha \frac{\partial E}{\partial W_{ij}^k}$$

误差

$$E=\frac{(y-\hat{y})^2}{2}$$

前向反馈

$$
h_1=\phi (\sum^2_i x_iW^1_{i1})\\
h_2=\phi (\sum^2_i x_iW^1_{i2})\\
h_3=\phi (\sum^2_i x_iW^1_{i3})
$$

$$
\hat{y}=\sum_i h_iW_i^2
$$

计算W:

$$
\Delta W_{ij}=-\alpha \frac{\partial E}{\partial W_{ij}}=-\alpha \frac{\partial(\frac{(y-\hat{y})^2}{2})}{\partial W_{ij}}=\alpha (y-\hat{y})\frac{\partial \hat{y}}{\partial W_{ij}}
$$

计:

$$
\delta_{ij}=\frac{\partial \hat{y}}{\partial W_{ij}}
$$

第二层w:

$$
\frac{\partial \hat{y}}{\partial W_i^2}=\frac{\partial(\sum_i^3 h_iW_i^2)}{\partial W_i^2}=h_i
$$

类似:

$$
\delta_1=h_1\\
\delta_2=h_2\\
\delta_3=h_3
$$

第二层权值更新:

$$
\Delta W_1^2=\alpha (y-\hat{y})h_1\\
\Delta W_2^2=\alpha (y-\hat{y})h_2\\
\Delta W_3^2=\alpha (y-\hat{y})h_3
$$

第一层权值更新:

$$
\frac{\partial \hat{y}}{\partial W^1_{ij}}=\sum^3_{p=1}\frac{\partial \hat{y}}{\partial h_p}\frac{\partial h_p}{\partial W^1_{ij}}
$$

第一个导数:

$$
\frac{\partial y}{\partial h_j}=\frac{\partial\sum^3_{i=1}(h_iW_i^2)}{\partial h_j}=W_j^2
$$

第二个导数:

$$
\frac{\partial h_i}{\partial W_{ij}^1}=\frac{\partial \phi_j(\sum_{i=1}^2(x_i W_{ij}^1))}{\partial(\sum_{i=1}^2(x_iW_{ij}^1))}\times\frac{\partial(\sum^2_{i=1}(x_iW^1_{ij}))}{\partial W^1_{ij}}
$$

结果:

$$
\frac{\partial h_j}{\partial W_{ij}^1}=\phi_j'x_i
$$

第一层

$$
\delta_{ij}=\frac{\partial\hat{y}}{\partial W_{ij}^1}=W^2_j \cdot \phi_j' \cdot x_i
$$

更新公式:

$$
\Delta W_{ij}=\alpha\cdot(y-\hat{y})\cdot\delta_{ij}
$$
