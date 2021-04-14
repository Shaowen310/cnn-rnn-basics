# %%
from IPython import get_ipython

# %% [markdown]
# Credit: CS5242 Course Material
#
# # Introduction
#
# Implementation of the Recurrent Neural Network (RNN) and apply RNN on sentiment analysis task.
#
# For each layer we will implement a forward and a backward function. The forward function will receive inputs and will return the outputs of this layer(loss layer will be a little different), and the backward pass will receive upstream derivatives and inputs and will return gradients with respect to the inputs. Gradients for weights or bias will be stored in parameters in this layer:
#
# ```python
# class SomeLayer(Layer):
#     # some layer type inherited from Layer class
#     def __init__(self, params):
#         # set up specific layer parameters
#         # initialize variables for the layer weights
#         # initialize variables for storing the gradients
#         # initialize other necessary variables
#     def forward(self, inputs):
#         # Receive inputs and return output
#         # Do some computations ...
#         z = # ... some intermediate value
#         # Do some more computations ...
#         outputs = # the outputs
#         return outputs
#     def backward(self, in_grads, inputs):
#         # Receive derivative of loss with respect to outputs,
#         # and compute derivative with respect to inputs.
#         # Use values in cache to compute derivatives
#         out_grads = # Derivative of loss with respect to inputs
#         self.w_grad = # Derivative of loss with respect to self.weights
#         return out_grads
# ```
#
# After implementing a bunch of layers (i.e. `RNN Cell`, `RNN`, `Bidirectional RNN`) in this way, we will be able to easily combine them to build classifiers for various applications whose input are sequential data (e.g. Sentiment Analysis).
# %% [markdown]
# # RNN Cell Layer
#
# RNN cell is the basic building block of RNN, which implements the specific operation at each time step of RNN. It has an hidden states of dimension `H` and accepts inputs of dimension `D`. Implement a simple type of RNN cell, formulated as follows:
#
# \begin{equation*}
# y=tanh(Wx+Uh+b),
# \end{equation*}
#
# where `x` and `h` are the inputs and hidden states respectively, and `W`, `U` and `b` are trainable kernel, recurrent_kernel and bias respectively.
# %% [markdown]
# ## Forward
#
# Please implement the function `RNNCell.forward(self, inputs)` and test the implementation using the following code. (`inputs` is a list of two numpy arrays, `[x, h]`).

# %%
import numpy as np
import keras
from keras import layers
import importlib
import rnn_layers

importlib.reload(rnn_layers)
from rnn_layers import RNNCell
from utils.tools import rel_error

N, D, H = 3, 10, 4
x = np.random.uniform(size=(N, D))
prev_h = np.random.uniform(size=(N, H))

rnn_cell = RNNCell(in_features=D, units=H)
out = rnn_cell.forward([x, prev_h])
# compare with the keras implementation
keras_x = layers.Input(shape=(1, D), name='x')
keras_prev_h = layers.Input(shape=(H, ), name='prev_h')
keras_rnn = layers.RNN(layers.SimpleRNNCell(H), name='rnn')(keras_x, initial_state=keras_prev_h)
keras_model = keras.Model(inputs=[keras_x, keras_prev_h], outputs=keras_rnn)
keras_model.get_layer('rnn').set_weights(
    [rnn_cell.kernel, rnn_cell.recurrent_kernel, rnn_cell.bias])
keras_out = keras_model.predict_on_batch([x[:, None, :], prev_h])

print('Relative error (<1e-5 will be fine): {}'.format(rel_error(keras_out, out)))

# %% [markdown]
# ## Backward
#
# Please implement the function `RNNCell.backward(self, in_grads, inputs)` and test the implementation using the following code. We need to compute the gradients to both the inputs and hidden states, as well as those trainable weights.

# %%
import numpy as np
import importlib
import rnn_layers

importlib.reload(rnn_layers)
from rnn_layers import RNNCell
from utils.check_grads import check_grads_layer

N, D, H = 3, 10, 4
x = np.random.uniform(size=(N, D))
prev_h = np.random.uniform(size=(N, H))
in_grads = np.random.uniform(size=(N, H))

rnn_cell = RNNCell(in_features=D, units=H)
check_grads_layer(rnn_cell, [x, prev_h], in_grads)

# %% [markdown]
# Then improve the implementation of RNN cell so that it can properly handle `NaN` input, and test it with the following code. **The gradients to those `NaN` input units are supposed to be zeros.**

# %%
import numpy as np
import importlib
import rnn_layers

importlib.reload(rnn_layers)
from rnn_layers import RNNCell
from utils.check_grads import check_grads_layer

N, D, H = 3, 10, 4
x = np.random.uniform(size=(N, D))
# set part of input to NaN
# this situation will be encountered in the following work
x[1:, :] = np.nan
prev_h = np.random.uniform(size=(N, H))
in_grads = np.random.uniform(size=(N, H))

rnn_cell = RNNCell(in_features=D, units=H)
check_grads_layer(rnn_cell, [x, prev_h], in_grads)

# %% [markdown]
# # RNN Layer
#
# RNN layer wraps any type of RNN cell so that it can operate over a sequence of input data of different length. In particular, it runs a instance of RNN cell over the inputs, holds and updates the hidden states for the RNN cell.
# %% [markdown]
# ## Forward
#
# Please implement the function `RNN.forward(self, inputs)` and test the implementation using the following code. Since NN layers generally proceed on a batch of data simultaneously, and for RNN, each input data may have different length, we define the input data format as an array of `(N, T, D)`, where `N` is the number of samples in a batch, `T` is the maximum length of input sequences, and `D` is the dimension of features at each time step. `NaN` is used to pad input sequences of different lenghts, so that the resulting length equals to `T`, e.g. `(x1, x2, ..., xk, NaN, NaN)`.

# %%
import numpy as np
import keras
from keras import layers
import importlib
import rnn_layers

importlib.reload(rnn_layers)
from rnn_layers import RNNCell, RNN
from utils.tools import rel_error

N, T, D, H = 2, 3, 4, 5
x = np.random.uniform(size=(N, T, D))
x[0, -1:, :] = np.nan
x[1, -2:, :] = np.nan
h0 = np.random.uniform(size=(H, ))

rnn_cell = RNNCell(in_features=D, units=H)
rnn = RNN(rnn_cell, h0=h0)
out = rnn.forward(x)

keras_x = layers.Input(shape=(T, D), name='x')
keras_h0 = layers.Input(shape=(H, ), name='h0')
keras_rnn = layers.RNN(layers.SimpleRNNCell(H), return_sequences=True,
                       name='rnn')(keras_x, initial_state=keras_h0)
keras_model = keras.Model(inputs=[keras_x, keras_h0], outputs=keras_rnn)
keras_model.get_layer('rnn').set_weights([rnn.kernel, rnn.recurrent_kernel, rnn.bias])
keras_out = keras_model.predict_on_batch([x, np.tile(h0, (N, 1))])

print('Relative error (<1e-5 will be fine): {}'.format(rel_error(keras_out, out)))

# %% [markdown]
# ## Backward
#
# Please implement the function `RNN.backward(self, in_grads, inputs)` and test the implementation using the following code (**note the internal gradients passed from next time steps**). Once again: the gradients to those `NaN` input units are supposed to be zeros

# %%
import numpy as np
import importlib
import rnn_layers

importlib.reload(rnn_layers)
from rnn_layers import RNNCell, RNN
from utils.check_grads import check_grads_layer

N, T, D, H = 2, 3, 4, 5
x = np.random.uniform(size=(N, T, D))
x[0, -1:, :] = np.nan
x[1, -2:, :] = np.nan
in_grads = np.random.uniform(size=(N, T, H))

rnn_cell = RNNCell(in_features=D, units=H)
rnn = RNN(rnn_cell)
check_grads_layer(rnn, x, in_grads)

# %% [markdown]
# # Bi-directional RNN Layer
#
# Vallina RNN operates over input sequence in one direction, so it has limitations as the future input information cannot be reached from the current state. On the contrary, Bi-directional RNN addresses this shortcoming by operating the input sequence in both forward and backward directions.
#
# Usually, Bi-directional RNN is implemented by running two independent RNNs in opposite direction of input data, and concatenating the outputs of the two RNNs. A useful function that can reverse a batch of sequence data is provided.
#
# ```python
# def _reverse_temporal_data(self, x, mask):
#     num_nan = np.sum(~mask, axis=1)
#     reversed_x = np.array(x[:, ::-1, :])
#     for i in range(num_nan.size):
#         reversed_x[i] = np.roll(reversed_x[i], x.shape[1]-num_nan[i], axis=0)
#     return reversed_x
# ```
# %% [markdown]
# ## Forward
#
# We provided the function `BidirectionalRNN.forward(self, inputs)` and the following code for testing. Note that `H` is the dimension of the hidden states of one internal RNN, so the actual dimension of the hidden states (or outputs) of Bidirectional RNN is `2*H`.

# %%
import numpy as np
import keras
from keras import layers
import importlib
import rnn_layers

importlib.reload(rnn_layers)
from rnn_layers import RNNCell, BidirectionalRNN
from utils.tools import rel_error

N, T, D, H = 2, 3, 4, 5
x = np.random.uniform(size=(N, T, D))
x[0, -1:, :] = np.nan
x[1, -2:, :] = np.nan
h0 = np.random.uniform(size=(N, H))
hr = np.random.uniform(size=(N, H))

rnn_cell = RNNCell(in_features=D, units=H)
brnn = BidirectionalRNN(rnn_cell, h0=h0, hr=hr)
out = brnn.forward(x)

keras_x = layers.Input(shape=(T, D), name='x')
keras_h0 = layers.Input(shape=(H, ), name='h0')
keras_hr = layers.Input(shape=(H, ), name='hr')
keras_x_masked = layers.Masking(mask_value=0.)(keras_x)
keras_rnn = layers.RNN(layers.SimpleRNNCell(H), return_sequences=True)
keras_brnn = layers.Bidirectional(keras_rnn, merge_mode='concat',
                                  name='brnn')(keras_x_masked, initial_state=[keras_h0, keras_hr])
keras_model = keras.Model(inputs=[keras_x, keras_h0, keras_hr], outputs=keras_brnn)
keras_model.get_layer('brnn').set_weights([
    brnn.forward_rnn.kernel, brnn.forward_rnn.recurrent_kernel, brnn.forward_rnn.bias,
    brnn.backward_rnn.kernel, brnn.backward_rnn.recurrent_kernel, brnn.backward_rnn.bias
])
keras_out = keras_model.predict_on_batch([np.nan_to_num(x), h0, hr])
nan_indices = np.where(np.any(np.isnan(x), axis=2))
keras_out[nan_indices[0], nan_indices[1], :] = np.nan

print('Relative error (<1e-5 will be fine): {}'.format(rel_error(keras_out, out)))

# %% [markdown]
# ## Backward
#
# Please refer to the provided forward function and implement the function `BidirectionalRNN.backward(self, inputs)`. Test the implementation using the following code.

# %%
import numpy as np
import importlib
import rnn_layers

importlib.reload(rnn_layers)
from rnn_layers import RNNCell, BidirectionalRNN
from utils.check_grads import check_grads_layer

N, T, D, H = 2, 3, 4, 5
x = np.random.uniform(size=(N, T, D))
x[0, -1:, :] = np.nan
x[1, -2:, :] = np.nan
in_grads = np.random.uniform(size=(N, T, H * 2))

rnn_cell = RNNCell(in_features=D, units=H)
brnn = BidirectionalRNN(rnn_cell)
check_grads_layer(brnn, x, in_grads)

# %% [markdown]
# # Sentiment Analysis using RNNs
#
# The dataset, `data/corpus.csv`, consists of 800 real movie comments and the corresponding labels that indicate whether the comments are positive or negative. For example:
# ```
# POSTIVE: I absolutely LOVE Harry Potter
# NEGATIVE: My dad's being stupid about brokeback mountain...
# ```
#
# We provide a basic NN for the experiments, which can be found in `applications.py`. The architecture is as follow:
# ```python
# FCLayer(vocab_size, 200, name='embedding')
# BidirectionalRNN(RNNCell(in_features=200, units=50))
# FCLayer(100, 32, name='fclayer1')
# TemporalPooling()
# FCLayer(32, 2, name='fclayer2')
# ```
# The input to the network is sequences of one-hot vectors, each of which represents a word. The 1st FC layer works as an [embedding layer](https://www.tensorflow.org/versions/master/programmers_guide/embedding) to learn and retrieve the word embedding vectors. After a Bi-directional RNN layer and another FC layer, a TemporalPooling layer (see `layers.py`) is used to mean-pooling a sequence of vectors into one vector, which will ignore the filling `NaN`s. The rest of the network is same as a normal NN classifier.

# %%
from utils import datasets
from applications import SentimentNet
from loss import SoftmaxCrossEntropy, L2
from optimizers import Adam
import numpy as np

np.random.seed(5242)

dataset = datasets.Sentiment()
model = SentimentNet(dataset.dictionary)
loss = SoftmaxCrossEntropy(num_class=2)

adam = Adam(lr=0.001, decay=0, scheduler_func=lambda lr, it: lr * 0.5 if it % 500 == 0 else lr)
model.compile(optimizer=adam, loss=loss, regularization=L2(w=0.001))
train_results, val_results, test_results = model.train(dataset,
                                                       train_batch=100,
                                                       val_batch=100,
                                                       test_batch=100,
                                                       epochs=50,
                                                       val_intervals=100,
                                                       test_intervals=300,
                                                       print_intervals=5)

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

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
