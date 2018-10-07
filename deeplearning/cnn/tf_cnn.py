import tensorflow as tf

# 输出的深度
k_output = 64

# 图片的参数
image_width = 10
image_height = 10
color_channels = 3

# 过滤器的宽度和高度
filter_size_width = 5
filter_size_height = 5

# 图片输入
image_input = tf.placeholder(tf.float32,
                             shape=[None, image_height, image_width, color_channels])

# 初始化卷积层的参数,权值和偏差
weight = tf.Variable(tf.truncated_normal([filter_size_height,
                                          filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# 定义卷积层
conv_layer = tf.nn.conv2d(image_input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer)   # 添加偏置
conv_layer = tf.nn.relu(conv_layer)       # 激活

# 最大池化

