# %%
from IPython import get_ipython

# %% [markdown]
# Credit: CS5242 Course Material
#
# # Introduction
#
# Implementation of the convolutional neural network (CNN) and apply CNN on the MNIST dataset.
#
# %% [markdown]
# # ReLU layer
# A convolution layer is usually followed by a non-linear activation function. We will provide the functions `forward` and `backward` of class `ReLU` in `layers.py` as an implementation example. Read through the function code and make sure you understand the derivation. Besides, we will explain the implementation of `ReLU` after `Convolution` using formula. You need to write down other layers' formulations in your reports.
#
# ## Forward Formulation
# Given input $x \in R^{B \times C \times H \times W}$ ($B$:batch size, $C$: number of channel, $H$: input height, $W$: input width),  output $y \in R^{B \times C \times H \times W}$ will be caculated like this:
#
# \begin{equation*}
# y=indicator(x) \times x
# \end{equation*}
#
# Here, $indicator(x)$ return the same size of input $x$, comparing $x$ with 0 element-wisely. If $x_{i,j,k,l} \geq 0$ return $z_{i,j,k,l}=1$. And the multiplication is also element-wise. If the input $x$ has only 2 dimensions, i.e. the batch dimension and the feature dimension, e.g. after the FC layer, the subscripts $j,k,l$ in the formula are merged into one $j$.
#
# ## Backward Formulation
# Given input $x \in R^{B \times C \times H \times W}$ ($B$:batch size, $C$: number of channel, $H$: input height, $W$: input width) and gradients to output of this layer $dy \in R^{B \times C \times H \times W}$, gradients to input $dx$ will be caculated like this:
#
# \begin{equation*}
# dx=indicator(x) \times dy
# \end{equation*}
# %% [markdown]
# # Covolution Layer
# In the file `layers.py`, the class `Convolution` will be initialized with `conv_params`, `initializer` and `name`, shown as below:
#
# ```python
#
# def __init__(self, conv_params, initializer=Guassian(), name='conv'):
#         super(Convolution, self).__init__(name=name)
#         self.trainable = True
#         self.kernel_h = conv_params['kernel_h'] # height of kernel
#         self.kernel_w = conv_params['kernel_w'] # width of kernel
#         self.pad = conv_params['pad']
#         self.stride = conv_params['stride']
#         self.in_channel = conv_params['in_channel']
#         self.out_channel = conv_params['out_channel']
#
#         self.weights = initializer.initialize((self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
#         self.bias = np.zeros((self.out_channel))
#
#         self.w_grad = np.zeros(self.weights.shape)
#         self.b_grad = np.zeros(self.bias.shape)
# ```
#
# `conv_params` is a dictionary, containing these parameters:
#
# - 'kernel_h': The height of kernel.
# - 'kernel_w': The width of kernel.
# - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
# - 'pad': The number of pixels padded to the bottom, top, left and right of each feature map. **Here, `pad=2` means a 2-pixel border of padded with zeros. So the total number of zeros for horizontal (or vertical) direction is 2\*pad=4**.
# - 'in_channel': The number of input channels.
# - 'out_channel': The number of output channels.
#
# `initializer` is an instance of Initializer class (leave it out right now)
# %% [markdown]
# ## Forward
# In the file `layers.py`, implement the forward pass for a convolutional layer in the function `forward` of class `Convolution`.
#
# The input consists of N data points, each with C channels, height H and width W. We convolve each input with K different kernels, where each filter spans all C channels and has height HH and width WW.
#
# Input:
#
# - inputs: Input data of shape (N, C, H, W)
#
# Test the implementation by restarting jupyter notebook kernel and running the following:

# %%
import numpy as np
from layers import Convolution
from utils.tools import rel_error

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import warnings

warnings.filterwarnings('ignore')

inputs = np.random.uniform(size=(10, 3, 30, 30))
params = {
    'kernel_h': 5,
    'kernel_w': 5,
    'pad': 0,
    'stride': 2,
    'in_channel': inputs.shape[1],
    'out_channel': 64,
}
layer = Convolution(params)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.Conv2D(filters=params['out_channel'],
                            kernel_size=(params['kernel_h'], params['kernel_w']),
                            strides=(params['stride'], params['stride']),
                            padding='valid',
                            data_format='channels_first',
                            input_shape=inputs.shape[1:])
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
weights = np.transpose(layer.weights, (2, 3, 1, 0))
keras_layer.set_weights([weights, layer.bias])
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))

# %% [markdown]
# ## Backward
# Implement the backward pass for the convolution operation in the function `backward` of `Convolution` class in the file `layers.py`.
#
# Test the implementation by restarting jupyter notebook and running the gradient check.
#
# In gradient checking, to get an approximate gradient for a parameter, we vary that parameter by a small amount (while keeping rest of parameters constant) and note the difference in the network loss. Dividing the difference in network loss by the amount we varied the parameter gives us an approximation for the gradient. We repeat this process for all the other parameters to obtain our numerical gradient.
#
# More links on gradient checking:
#
# http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
#
# https://www.coursera.org/learn/machine-learning/lecture/Y3s6r/gradient-checking

# %%
from layers import Convolution
import numpy as np
from utils.check_grads import check_grads_layer

batch = 10
conv_params = {
    'kernel_h': 3,
    'kernel_w': 3,
    'pad': 0,
    'stride': 2,
    'in_channel': 3,
    'out_channel': 10
}
in_height = 10
in_width = 20
out_height = 1 + (in_height + 2 * conv_params['pad'] -
                  conv_params['kernel_h']) // conv_params['stride']
out_width = 1 + (in_width + 2 * conv_params['pad'] -
                 conv_params['kernel_w']) // conv_params['stride']
inputs = np.random.uniform(size=(batch, conv_params['in_channel'], in_height, in_width))
in_grads = np.random.uniform(size=(batch, conv_params['out_channel'], out_height, out_width))
conv = Convolution(conv_params)
check_grads_layer(conv, inputs, in_grads)

# %% [markdown]
# # Dropout Layer
# Dropout [1] is a technique for regularizing neural networks by randomly setting some features to zero during the forward pass.
#
# [1] Geoffrey E. Hinton et al, "Improving neural networks by preventing co-adaptation of feature detectors", arXiv 2012
#
# In the file `layers.py`, the class `FCLayer` will be initialized with `ratio`, `seed` and `name`, shown as below:
# ```python
# def __init__(self, ratio, name='dropout', seed=None):
#         super(Dropout, self).__init__(name=name)
#         self.ratio = ratio
#         self.mask = None
#         self.seed = seed
# ```
#
# - `ratio`: The probability of setting a neuron to zero
# - `seed`: Random seed to sample from inputs, so as to get mask. (default as None)
# %% [markdown]
# ## Forward
# In the file `layers.py`, implement the forward pass for dropout. Since dropout behaves differently during training and testing, make sure to implement the operation for both modes.  `p` refers to the probability of setting a neuron to zero. We will follow the Caffe convention where we multiply the outputs by `1/(1-p)` during training.
# %% [markdown]
# ## Backward
# In the file `layers.py`, implement the backward pass for dropout. Check the implementation by restarting jupyter notebook and running the following cell.

# %%
from layers import Dropout
import numpy as np
from utils.check_grads import check_grads_layer

ratio = 0.1
height = 10
width = 20
channel = 10
np.random.seed(1234)
inputs = np.random.uniform(size=(batch, channel, height, width))
in_grads = np.random.uniform(size=(batch, channel, height, width))
dropout = Dropout(ratio, seed=1234)
dropout.set_mode(True)
check_grads_layer(dropout, inputs, in_grads)

# %% [markdown]
# # Pooling Layer
# In the file `layers.py`, the class `Pooling` will be initialized with `pool_params`, and `name`, shown as below:
# ```python
# def __init__(self, pool_params, name='pooling'):
#         super(Pooling, self).__init__(name=name)
#         self.pool_type = pool_params['pool_type']
#         self.pool_height = pool_params['pool_height']
#         self.pool_width = pool_params['pool_width']
#         self.stride = pool_params['stride']
#         self.pad = pool_params['pad']
# ```
#
# `pool_params` is a dictionary, containing these parameters:
#
# - 'pool_type': The type of pooling, 'max' or 'avg'
# - 'pool_h': The height of pooling kernel.
# - 'pool_w': The width of pooling kernel.
# - 'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
# - 'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. **Here, `pad=2` means a 2-pixel border of padding with zeros**.
# %% [markdown]
# ## Forward
# Implement the forward pass for the pooling operation in the function `forward` of class `Pooling` in the file `layers.py`.
#
# Test the implementation by restarting jupyter notebook kernel and running the following:

# %%
import numpy as np
from layers import Pooling
from utils.tools import rel_error

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import warnings

warnings.filterwarnings('ignore')

inputs = np.random.uniform(size=(10, 3, 30, 30))
params = {
    'pool_type': 'max',
    'pool_height': 5,
    'pool_width': 5,
    'pad': 0,
    'stride': 2,
}
layer = Pooling(params)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.MaxPooling2D(pool_size=(params['pool_height'], params['pool_width']),
                                  strides=params['stride'],
                                  padding='valid',
                                  data_format='channels_first',
                                  input_shape=inputs.shape[1:])
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))

# %% [markdown]
# ## Backward
# Implement the backward pass for the max-pooling operation in the function `backward` of class `Pooling` in the file `layers.py`.
#
# Please make sure having implemented both ’max’ and ’avg’ pooling.
# %% [markdown]
# # FC Layer
#
# FC layer (short for fully connected layer) is also called linear layer or dense layer.
#
# In the file `layers.py`, the class `FCLayer` will be initialized with `in_features`, `out_features`, and `name`, shown as below:
# ```python
# def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
#         super(FCLayer, self).__init__(name=name)
#         self.trainable = True
#         self.weights = initializer.initialize((in_features, out_features))
#         self.bias = initializer.initialize(out_features)
#
#         self.w_grad = np.zeros(self.weights.shape)
#         self.b_grad = np.zeros(self.bias.shape)
# ```
#
# - `in_features`: The number of inputs features
# - `out_features`: The numbet of required outputs features
# %% [markdown]
# ## Forward
# Implement the forward pass for the pooling operation in the function `forward` of class `FCLayer` in the file `layers.py`.
#
# Test the implementation by restarting jupyter notebook kernel and running the following:

# %%
import numpy as np
from layers import FCLayer
from utils.tools import rel_error

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import warnings

warnings.filterwarnings('ignore')

inputs = np.random.uniform(size=(10, 20))

layer = FCLayer(in_features=inputs.shape[1], out_features=100)
out = layer.forward(inputs)

keras_model = keras.Sequential()
keras_layer = layers.Dense(100,
                           input_shape=inputs.shape[1:],
                           use_bias=True,
                           kernel_initializer='random_normal',
                           bias_initializer='zeros')
# print (len(keras_layer.get_weights()))
keras_model.add(keras_layer)
sgd = optimizers.SGD(lr=0.01)
keras_model.compile(loss='mean_squared_error', optimizer='sgd')
keras_layer.set_weights([layer.weights, layer.bias])
keras_out = keras_model.predict(inputs, batch_size=inputs.shape[0])
print('Relative error (<1e-6 will be fine): ', rel_error(out, keras_out))

# %% [markdown]
# ## Backward
# Implement the backward pass for the max-pooling operation in the function `backward` of class `FCLayer` in the file `layers.py`.
# %% [markdown]
# # SoftmaxCrossEntropy Loss
# We write Softmax and CrossEntropy together because it can avoid some numeric overflow problem.In the file `loss.py`, the class `SoftmaxCrossEntropy` will be initialized with `num_class`,  shown as below:
# ```python
# def __init__(self, num_class):
#         super(SoftmaxCrossEntropy, self).__init__()
#         self.num_class = num_class
# ```
#
# `num_class`: The number of category
# %% [markdown]
# ## Forward
# Implement the forward pass for the pooling operation in the function `forward` of class `FCLayer` in the file `layers.py`.
#
# Test your implementation by restarting jupyter notebook kernel and running the following:
# %%
import numpy as np
from loss import SoftmaxCrossEntropy
from utils.tools import rel_error

import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import backend as K

import warnings

warnings.filterwarnings('ignore')

batch = 10
num_class = 10
inputs = np.random.uniform(size=(batch, num_class))
targets = np.random.randint(num_class, size=batch)

loss = SoftmaxCrossEntropy(num_class)
out, _ = loss.forward(inputs, targets)

keras_inputs = K.softmax(inputs)
keras_targets = np.zeros(inputs.shape, dtype='int')
for i in range(batch):
    keras_targets[i, targets[i]] = 1
keras_out = K.mean(K.categorical_crossentropy(keras_targets, keras_inputs, from_logits=False))
print('Relative error (<1e-6 will be fine): ', rel_error(out, K.eval(keras_out)))

# %% [markdown]
# ## Backward
# In the file `loss.py`, implement the backward pass for `SodtmaxCrossEntropy`.
# %% [markdown]
# # Optimizer
# In the file `optimizers.py`, there are 4 types of optimizer (`SGD`, `Adam`, `RMSprop` and `Adagrad`). You only need to implement the `update` function of `SGD`(mini-batch SGD with momentum) and `Adam`. These two types of optimizers are initialized like this:
#
# ```python
# class SGD(Optimizer):
#     def __init__(self, lr=0.01, momentum=0, decay=0, sheduler_func = None):
#         super(SGD, self).__init__(lr)
#         self.momentum = momentum
#         self.moments = None
#         self.decay = decay
#         self.sheduler_func = sheduler_func
#
# class Adam(Optimizer):
#     def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, sheduler_func=None):
#         super(Adam, self).__init__(lr)
#         self.beta_1 = beta_1
#         self.beta_2 = beta_2
#         self.epsilon = epsilon
#         self.decay = decay
#         if not self.epsilon:
#             self.epsilon = 1e-8
#         self.moments = None
#         self.accumulators = None
#         self.sheduler_func = sheduler_func
# ```
#
# For Both optimizers:
# - `lr`: The initial learning rate.
# - `decay`: The learning rate decay ratio
# - `sheduler_func`: Function to change learning rate with respect to iterations
#
# For `SGD`:
# - `momentum`: The ratio of moments
#
#
# For `Adam`:
# More details can be seen in reference.
#
# **reference:**
# http://cs231n.github.io/neural-networks-3/#update
# %% [markdown]
# # Train the net on full MNIST data
# By training the `MNISTNet` for one epoch, it should achieve about 90% on the validation and test set.

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from applications import MNISTNet
from loss import SoftmaxCrossEntropy, L2
from optimizers import Adam
from utils.datasets import MNIST
import numpy as np

mnist = MNIST()
mnist.load()
idx = np.random.randint(mnist.num_train, size=4)
print('\nFour examples of training images:')
img = mnist.x_train[idx][:, 0, :, :]

plt.figure(1, figsize=(18, 18))
plt.subplot(1, 4, 1)
plt.imshow(img[0])
plt.subplot(1, 4, 2)
plt.imshow(img[1])
plt.subplot(1, 4, 3)
plt.imshow(img[2])
plt.subplot(1, 4, 4)
plt.imshow(img[3])

# %%
model = MNISTNet()
loss = SoftmaxCrossEntropy(num_class=10)


# define your learning rate sheduler
def func(lr, iteration):
    if iteration % 1000 == 0:
        return lr * 0.5
    else:
        return lr


def func_more_freq_decrease(lr, iteration):
    if iteration % 200 == 0:
        return lr * 0.8
    else:
        return lr


adam = Adam(lr=0.005, decay=0, sheduler_func=func_more_freq_decrease)
l2 = L2(w=0.001)  # L2 regularization with lambda=0.001
model.compile(optimizer=adam, loss=loss, regularization=l2)
train_results, val_results, test_results = model.train(mnist,
                                                       train_batch=30,
                                                       val_batch=1000,
                                                       test_batch=1000,
                                                       epochs=2,
                                                       val_intervals=100,
                                                       test_intervals=300,
                                                       print_intervals=100)

# %%
plt.figure(2, figsize=(18, 8))
plt.subplot(2, 3, 1)
plt.title('Training loss')
plt.plot(train_results[:, 0], train_results[:, 1])
plt.subplot(2, 3, 4)
plt.title('Training accuracy')
plt.plot(train_results[:, 0], train_results[:, 2])
plt.subplot(2, 3, 2)
plt.title('Validation loss')
plt.plot(val_results[:, 0], val_results[:, 1])
plt.subplot(2, 3, 5)
plt.title('Validation accuracy')
plt.plot(val_results[:, 0], val_results[:, 2])
plt.subplot(2, 3, 3)
plt.title('Testing loss')
plt.plot(test_results[:, 0], test_results[:, 1])
plt.subplot(2, 3, 6)
plt.title('Testing accuracy')
plt.plot(test_results[:, 0], test_results[:, 2])

# %% [markdown]
# ## Change of learning rate
# If we change the initial learning rate from 0.001 to 0.1, the training process becomes unstable and the loss is out of control. Thus, we need to be careful when setting the initial learning rate.

# %%
model = MNISTNet()
loss = SoftmaxCrossEntropy(num_class=10)


# define your learning rate sheduler
def func(lr, iteration):
    if iteration % 1000 == 0:
        return lr * 0.5
    else:
        return lr


adam = Adam(lr=0.01, decay=0, sheduler_func=func)
l2 = L2(w=0.001)  # L2 regularization with lambda=0.001
model.compile(optimizer=adam, loss=loss, regularization=l2)
train_results, val_results, test_results = model.train(mnist,
                                                       train_batch=30,
                                                       val_batch=1000,
                                                       test_batch=1000,
                                                       epochs=2,
                                                       val_intervals=100,
                                                       test_intervals=300,
                                                       print_intervals=100)

# %%
plt.figure(2, figsize=(18, 8))
plt.subplot(2, 3, 1)
plt.title('Training loss')
plt.plot(train_results[:, 0], train_results[:, 1])
plt.subplot(2, 3, 4)
plt.title('Training accuracy')
plt.plot(train_results[:, 0], train_results[:, 2])
plt.subplot(2, 3, 2)
plt.title('Validation loss')
plt.plot(val_results[:, 0], val_results[:, 1])
plt.subplot(2, 3, 5)
plt.title('Validation accuracy')
plt.plot(val_results[:, 0], val_results[:, 2])
plt.subplot(2, 3, 3)
plt.title('Testing loss')
plt.plot(test_results[:, 0], test_results[:, 1])
plt.subplot(2, 3, 6)
plt.title('Testing accuracy')
plt.plot(test_results[:, 0], test_results[:, 2])

# %%
