"""
change log:
- Version 1: change the out_grads of `backward` function of `ReLU` layer into inputs_grads instead of in_grads
"""

import numpy as np
from utils.tools import *
from utils.img2cols import *

class Layer(object):
    """

    """

    def __init__(self, name):
        """Initialization"""
        self.name = name
        self.training = True  # The phrase, if for training then true
        self.trainable = False  # Whether there are parameters in this layer that can be trained

    def forward(self, inputs):
        """Forward pass, reture outputs"""
        raise NotImplementedError

    def backward(self, in_grads, inputs):
        """Backward pass, return gradients to inputs"""
        raise NotImplementedError

    def update(self, optimizer):
        """Update parameters in this layer"""
        pass

    def set_mode(self, training):
        """Set the phrase/mode into training (True) or tesing (False)"""
        self.training = training

    def set_trainable(self, trainable):
        """Set the layer can be trainable (True) or not (False)"""
        self.trainable = trainable

    def get_params(self, prefix):
        """Reture parameters and gradients of this layer"""
        return None


class FCLayer(Layer):
    def __init__(self, in_features, out_features, name='fclayer', initializer=Guassian()):
        """Initialization

        # Arguments
            in_features: int, the number of inputs features
            out_features: int, the numbet of required outputs features
            initializer: Initializer class, to initialize weights
        """
        super(FCLayer, self).__init__(name=name)
        self.trainable = True

        self.weights = initializer.initialize((in_features, out_features))
        self.bias = np.zeros(out_features)

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_features)

        # Returns
            outputs: numpy array with shape (batch, out_features)
        """

        outputs = inputs @ self.weights + self.bias
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_features), gradients to outputs
            inputs: numpy array with shape (batch, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_features), gradients to inputs
        """
        self.w_grad = inputs.T @ in_grads
        self.b_grad = np.sum(in_grads, axis=0)
        out_grads = in_grads @ self.weights.T
        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params

        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class Convolution(Layer):
    def __init__(self, conv_params, initializer=Guassian(), name='conv'):
        """Initialization

        # Arguments
            conv_params: dictionary, containing these parameters:
                'kernel_h': The height of kernel.
                'kernel_w': The width of kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels padded to the bottom, top, left and right of each feature map. Here, pad=2 means a 2-pixel border of padded with zeros.
                'in_channel': The number of input channels.
                'out_channel': The number of output channels.
            initializer: Initializer class, to initialize weights
        """
        super(Convolution, self).__init__(name=name)
        self.trainable = True
        self.kernel_h = conv_params['kernel_h']  # height of kernel
        self.kernel_w = conv_params['kernel_w']  # width of kernel
        self.pad = conv_params['pad']
        self.stride = conv_params['stride']
        self.in_channel = conv_params['in_channel']
        self.out_channel = conv_params['out_channel']

        self.weights = initializer.initialize(
            (self.out_channel, self.in_channel, self.kernel_h, self.kernel_w))
        self.bias = np.zeros((self.out_channel))

        self.w_grad = np.zeros(self.weights.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, out_channel, out_height, out_width)
        """
        batch, in_channel, in_height, in_width = inputs.shape

        out_height = (in_height + 2*self.pad - self.kernel_h) // self.stride + 1
        out_width = (in_width + 2*self.pad - self.kernel_w) // self.stride + 1

        # convert input image to columns, shape (in_channel * kernel_h * kernel_w, out_height * out_width * batch)
        inputs_cols = img2col_indices(inputs, self.kernel_h, self.kernel_w, self.pad, self.stride)
        # convert the kernels to rows, shape (out_channel, in_channel * kernel_h * kernel_w)
        weights_rows = self.weights.reshape((self.out_channel, -1))

        outputs = weights_rows @ inputs_cols + self.bias.reshape((-1, 1))

        outputs = outputs.reshape(self.out_channel, out_height, out_width, batch)
        outputs = outputs.transpose(3, 0, 1, 2)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass, store gradients to self.weights into self.w_grad and store gradients to self.bias into self.b_grad

        # Arguments
            in_grads: numpy array with shape (batch, out_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        self.b_grad = np.sum(in_grads, axis=(0, 2, 3))

        # convert in_grads to rows, shape (out_channel, out_height * out_width * batch)
        in_grads_rows = in_grads.transpose(1, 2, 3, 0).reshape((self.out_channel, -1))
        # convert input image to columns, shape (in_channel * kernel_h * kernel_w, out_height * out_width * batch)
        inputs_cols = img2col_indices(inputs, self.kernel_h, self.kernel_w, self.pad, self.stride)
        self.w_grad = in_grads_rows @ inputs_cols.T
        self.w_grad = self.w_grad.reshape(self.weights.shape)

        weights_rows = self.weights.reshape((self.out_channel, -1))
        out_grads_cols = weights_rows.T @ in_grads_rows

        out_grads = col2img_indices(out_grads_cols, inputs.shape, self.kernel_h, self.kernel_w, self.pad, self.stride)

        return out_grads

    def update(self, params):
        """Update parameters (self.weights and self.bias) with new params

        # Arguments
            params: dictionary, one key contains 'weights' and the other contains 'bias'

        # Returns
            none
        """
        for k, v in params.items():
            if 'weights' in k:
                self.weights = v
            else:
                self.bias = v

    def get_params(self, prefix):
        """Return parameters (self.weights and self.bias) as well as gradients (self.w_grad and self.b_grad)

        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer, one key contains 'weights' and the other contains 'bias'
            grads: dictionary, store gradients of this layer, one key contains 'weights' and the other contains 'bias'

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/weights': self.weights,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/weights': self.w_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class ReLU(Layer):
    def __init__(self, name='relu'):
        """Initialization
        """
        super(ReLU, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        outputs = np.maximum(0, inputs)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        inputs_grads = (inputs >= 0) * in_grads
        out_grads = inputs_grads
        return out_grads


# TODO: add padding
class Pooling(Layer):
    def __init__(self, pool_params, name='pooling'):
        """Initialization

        # Arguments
            pool_params is a dictionary, containing these parameters:
                'pool_type': The type of pooling, 'max' or 'avg'
                'pool_h': The height of pooling kernel.
                'pool_w': The width of pooling kernel.
                'stride': The number of pixels between adjacent receptive fields in the horizontal and vertical directions.
                'pad': The number of pixels that will be used to zero-pad the input in each x-y direction. Here, pad=2 means a 2-pixel border of padding with zeros.
        """
        super(Pooling, self).__init__(name=name)
        self.pool_type = pool_params['pool_type']
        self.pool_height = pool_params['pool_height']
        self.pool_width = pool_params['pool_width']
        self.stride = pool_params['stride']
        self.pad = pool_params['pad']

        self.max_idx = None

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel, out_height, out_width)
        """
        pool_func = None
        if self.pool_type == 'max':
            pool_func = maxpool
        elif self.pool_type == 'avg':
            pool_func = avgpool
        else:
            raise ValueError('Pool type not supported')

        batch, in_channel, in_height, in_width = inputs.shape

        out_height = (in_height - self.pool_height) // self.stride + 1
        out_width = (in_width - self.pool_width) // self.stride + 1

        # make inputs.shape same as convolution operation
        inputs_reshaped = inputs.reshape((batch * in_channel, 1, in_height, in_width))
        inputs_cols = img2col_indices(inputs_reshaped, self.pool_height, self.pool_width, self.pad, self.stride)

        outputs, self.max_idx = pool_func(inputs_cols)

        outputs = outputs.reshape(out_height, out_width, batch, in_channel)
        outputs = outputs.transpose(2, 3, 0, 1)

        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel, out_height, out_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs
        """
        dpool_func = None
        if self.pool_type == 'max':
            dpool_func = dmaxpool
        elif self.pool_type == 'avg':
            dpool_func = davgpool
        else:
            raise ValueError('Pool type is not supported')

        batch, in_channel, in_height, in_width = inputs.shape

        # make inputs.shape same as convolution operation
        inputs_reshaped = inputs.reshape(batch * in_channel, 1, in_height, in_width)
        inputs_cols = img2col_indices(inputs_reshaped, self.pool_height, self.pool_width, self.pad, self.stride)

        out_grads_cols = np.zeros_like(inputs_cols)
        in_grads_cols = in_grads.transpose(2, 3, 0, 1).ravel()

        out_grads = dpool_func(out_grads_cols, in_grads_cols, self.max_idx)

        out_grads = col2img_indices(out_grads_cols, (batch * in_channel, 1, in_height, in_width), self.pool_height, self.pool_width, self.pad, self.stride)
        out_grads = out_grads.reshape(inputs.shape)

        return out_grads

class Dropout(Layer):
    def __init__(self, ratio, name='dropout', seed=None):
        """Initialization

        # Arguments
            ratio: float [0, 1], the probability of setting a neuron to zero
            seed: int, random seed to sample from inputs, so as to get mask. (default as None)
        """
        super(Dropout, self).__init__(name=name)
        self.ratio = ratio
        self.mask = None
        self.seed = seed

    def forward(self, inputs):
        """Forward pass (Hint: use self.training to decide the phrase/mode of the model)

        # Arguments
            inputs: numpy array

        # Returns
            outputs: numpy array
        """
        if self.training:
            if self.seed is not None:
                np.random.seed(self.seed)
            self.mask = np.random.binomial(1, 1-self.ratio, size=inputs.shape) / (1-self.ratio)
            outputs = inputs * self.mask
        else:
            self.mask = np.ones(inputs.shape)
            outputs = inputs
        return outputs


    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array, gradients to outputs
            inputs: numpy array, same with forward inputs

        # Returns
            out_grads: numpy array, gradients to inputs 
        """
        out_grads = in_grads * self.mask
        return out_grads


class Flatten(Layer):
    def __init__(self, name='flatten', seed=None):
        """Initialization
        """
        super(Flatten, self).__init__(name=name)

    def forward(self, inputs):
        """Forward pass

        # Arguments
            inputs: numpy array with shape (batch, in_channel, in_height, in_width)

        # Returns
            outputs: numpy array with shape (batch, in_channel*in_height*in_width)
        """
        batch = inputs.shape[0]
        outputs = inputs.copy().reshape(batch, -1)
        return outputs

    def backward(self, in_grads, inputs):
        """Backward pass

        # Arguments
            in_grads: numpy array with shape (batch, in_channel*in_height*in_width), gradients to outputs
            inputs: numpy array with shape (batch, in_channel, in_height, in_width), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, in_channel, in_height, in_width), gradients to inputs 
        """
        out_grads = in_grads.copy().reshape(inputs.shape)
        return out_grads
