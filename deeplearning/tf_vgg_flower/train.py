#!/usr/bin/python
import os

import numpy as np
import tensorflow as tf

from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

data_dir = 'data/flower_photos'

contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

print("all classes", classes)
