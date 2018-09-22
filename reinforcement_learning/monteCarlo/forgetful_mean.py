#!/usr/bin/python

# 对于一般的均值来说，所有值对于均值来说都是等价的
# 在MC控制中，更新动作值的时候越新的动作值越重要，
# 早期探索阶段（以前的动作值）应该被遗忘
# 这里引入可遗忘的均值，对于越早的值，计算的时候对mean影响越小
import numpy as np

x = np.hstack((np.ones(10), 10 * np.ones(10)))

# alpha控制了算法的遗忘程度
aplha_values = np.arange(0, .3, .01) + .01

# 为了对比，下面是增量均值（传统的mean算法）
def running_mean(x):
    mu = 0
    mean_vals = []
    for k in range(0, len(x)):
        mu += (1.0 / (k + 1)) * (x[k] - mu)
        mean_vals.append(mu)
    return mean_vals

# 下面是遗忘均值算法
# alpha控制遗忘程度，alpha越高，均值对于后面的值越敏感
def forgetful_mean(x, alpha):
    mu = 0
    mean_vals = []
    for k in range(0, len(x)):
        mu += alpha * (x[k] - mu)
        mean_vals.append(mu)
    return mean_vals

if __name__ == "__main__":
    print("The x is:", x)
    print("The running mean function returns:", running_mean(x)[-1])
    print("The forgetful mean function returns:")
    for alpha in aplha_values:
        print(np.round(forgetful_mean(x, alpha)[-1], 4), \
                "(alpha={})".format(np.round(alpha, 2)))
