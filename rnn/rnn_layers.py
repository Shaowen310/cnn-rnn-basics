import numpy as np
from layers import Layer
from utils.tools import *
import copy

"""
This file defines layer types that are commonly used for recurrent neural networks.
"""

class RNNCell(Layer):
    def __init__(self, in_features, units, name='rnn_cell', initializer=Guassian()):
        """
        # Arguments
            in_features: int, the number of inputs features
            units: int, the number of hidden units
            initializer: Initializer class, to initialize weights
        """
        super(RNNCell, self).__init__(name=name)
        self.trainable = True

        self.kernel = initializer.initialize((in_features, units))
        self.recurrent_kernel = initializer.initialize((units, units))
        self.bias = np.zeros(units)

        self.kernel_grad = np.zeros(self.kernel.shape)
        self.r_kernel_grad = np.zeros(self.recurrent_kernel.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """
        # Arguments
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)]

        # Returns
            outputs: numpy array with shape (batch, units)
        """
        batch = len(inputs[0])
        inputs[0] @ self.kernel
        inputs[1] @ self.recurrent_kernel
        a_t = inputs[0] @ self.kernel + inputs[1] @ self.recurrent_kernel + np.repeat(self.bias.reshape((1, -1)), batch, axis=0)
        return np.tanh(a_t)

    def backward(self, in_grads, inputs):
        """
        # Arguments
            in_grads: numpy array with shape (batch, units), gradients to outputs
            inputs: [input numpy array with shape (batch, in_features), 
                    state numpy array with shape (batch, units)], same with forward inputs

        # Returns
            out_grads: [gradients to input numpy array with shape (batch, in_features), 
                        gradients to state numpy array with shape (batch, units)]
        """
        h_t = self.forward(inputs)
        d_a_t = (1 - np.square(h_t)) * in_grads
        d_a_t = np.nan_to_num(d_a_t)
        self.b_grad = np.sum(d_a_t, axis=0)
        self.kernel_grad = np.nan_to_num(inputs[0].T) @ d_a_t
        self.r_kernel_grad = np.nan_to_num(inputs[1].T) @ d_a_t
        out_grads_in = d_a_t @ self.kernel.T
        out_grads_h = d_a_t @ self.recurrent_kernel.T

        return [out_grads_in, out_grads_h]

    def update(self, params):
        """Update parameters with new params
        """
        for k,v in params.items():
            if '/kernel' in k:
                self.kernel = v
            elif '/recurrent_kernel' in k:
                self.recurrent_kernel = v
            elif '/bias' in k:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters and gradients
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/kernel': self.kernel,
                prefix+':'+self.name+'/recurrent_kernel': self.recurrent_kernel,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/kernel': self.kernel_grad,
                prefix+':'+self.name+'/recurrent_kernel': self.r_kernel_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None


class RNN(Layer):
    def __init__(self, cell, h0=None, name='rnn'):
        """
        # Arguments
            cell: instance of RNN Cell
            h0: default initial state, numpy array with shape (units,)
        """
        super(RNN, self).__init__(name=name)
        self.trainable = True
        self.cell = cell
        if h0 is None:
            self.h0 = np.zeros_like(self.cell.bias)
        else:
            self.h0 = h0
        
        self.kernel = self.cell.kernel
        self.recurrent_kernel = self.cell.recurrent_kernel
        self.bias = self.cell.bias

        self.kernel_grad = np.zeros(self.kernel.shape)
        self.r_kernel_grad = np.zeros(self.recurrent_kernel.shape)
        self.b_grad = np.zeros(self.bias.shape)

    def forward(self, inputs):
        """
        # Arguments
            inputs: input numpy array with shape (batch, time_steps, in_features), 

        # Returns
            outputs: numpy array with shape (batch, time_steps, units)
        """
        batch = inputs.shape[0]
        h_0 = self.h0
        if self.h0.size == self.cell.bias.size:
            h_0 = np.tile(self.h0, batch).reshape((batch, -1))

        inputs = inputs.transpose((1, 0, 2))
        outputs = np.empty((inputs.shape[0], inputs.shape[1], len(self.cell.bias)))
        
        h_t = h_0
        for t in range(0, len(inputs)):
            h_t = self.cell.forward([inputs[t], h_t])
            outputs[t] = h_t

        return outputs.transpose((1, 0, 2))

    def backward(self, in_grads, inputs):
        """
        # Arguments
            in_grads: numpy array with shape (batch, time_steps, units), gradients to outputs
            inputs: numpy array with shape (batch, time_steps, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, time_steps, in_features), gradients to inputs
        """
        batch = inputs.shape[0]
        h_0 = self.h0
        if self.h0.size == self.cell.bias.size:
            h_0 = np.tile(self.h0, batch).reshape((batch, -1))

        outputs = self.forward(inputs)

        inputs = inputs.transpose((1, 0, 2))
        outputs = outputs.transpose((1, 0, 2))

        in_grads = in_grads.transpose((1, 0, 2))
        out_grads = np.empty(inputs.shape)

        self.kernel_grad = np.zeros(self.kernel.shape)
        self.r_kernel_grad = np.zeros(self.recurrent_kernel.shape)
        self.b_grad = np.zeros(self.bias.shape)
        
        out_grads_s = np.zeros(in_grads.shape[1:3])
        for t in range(len(inputs)-1, -1, -1):
            out_grads_t = self.cell.backward(in_grads[t]+out_grads_s, [inputs[t], outputs[t-1] if t > 0 else h_0])
            self.kernel_grad += self.cell.kernel_grad
            self.r_kernel_grad += self.cell.r_kernel_grad
            self.b_grad += self.cell.b_grad
            (out_grads[t], out_grads_s) = out_grads_t

        return out_grads.transpose((1, 0, 2))

    def update(self, params):
        """Update parameters with new params
        """
        for k,v in params.items():
            if '/kernel' in k:
                self.kernel = v
            elif '/recurrent_kernel' in k:
                self.recurrent_kernel = v
            elif '/bias' in k:
                self.bias = v
        
    def get_params(self, prefix):
        """Return parameters and gradients
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/kernel': self.kernel,
                prefix+':'+self.name+'/recurrent_kernel': self.recurrent_kernel,
                prefix+':'+self.name+'/bias': self.bias
            }
            grads = {
                prefix+':'+self.name+'/kernel': self.kernel_grad,
                prefix+':'+self.name+'/recurrent_kernel': self.r_kernel_grad,
                prefix+':'+self.name+'/bias': self.b_grad
            }
            return params, grads
        else:
            return None        


class BidirectionalRNN(Layer):
    """ Bi-directional RNN in Concatenating Mode
    """
    def __init__(self, cell, h0=None, hr=None, name='brnn'):
        """Initialize two inner RNNs for forward and backward processes, respectively

        # Arguments
            cell: instance of RNN Cell(D, H) for initializing the two RNNs
            h0: default initial state for forward phase, numpy array with shape (units,)
            hr: default initial state for backward phase, numpy array with shape (units,)
        """
        super(BidirectionalRNN, self).__init__(name=name)
        self.trainable = True
        self.forward_rnn = RNN(cell, h0, 'forward_rnn')
        self.backward_rnn = RNN(copy.deepcopy(cell), hr, 'backward_rnn')

    def _reverse_temporal_data(self, x, mask):
        """ Reverse a batch of sequence data

        # Arguments
            x: a numpy array of shape (batch, time_steps, units), e.g.
                [[x_0_0, x_0_1, ..., x_0_k1, Unknown],
                ...
                [x_n_0, x_n_1, ..., x_n_k2, Unknown, Unknown]] (x_i_j is a vector of dimension of D)
            mask: a numpy array of shape (batch, time_steps), indicating the valid values, e.g.
                [[1, 1, ..., 1, 0],
                ...
                [1, 1, ..., 1, 0, 0]]

        # Returns
            reversed_x: numpy array with shape (batch, time_steps, units)
        """
        num_nan = np.sum(~mask, axis=1)
        reversed_x = np.array(x[:, ::-1, :])
        for i in range(num_nan.size):
            reversed_x[i] = np.roll(reversed_x[i], x.shape[1]-num_nan[i], axis=0)
        return reversed_x

    def forward(self, inputs):
        """
        Forward pass for concatenating hidden vectors obtained from the RNN 
        processing on normal sentences and the RNN processing on reversed sentences.
        Outputs concatenate the two produced sequences.

        # Arguments
            inputs: input numpy array with shape (batch, time_steps, in_features), 

        # Returns
            outputs: numpy array with shape (batch, time_steps, units*2)
        """
        mask = ~np.any(np.isnan(inputs), axis=2)
        forward_outputs = self.forward_rnn.forward(inputs)
        backward_outputs = self.backward_rnn.forward(self._reverse_temporal_data(inputs, mask))
        outputs = np.concatenate([forward_outputs, self._reverse_temporal_data(backward_outputs, mask)], axis=2)
        return outputs

    def backward(self, in_grads, inputs):
        """
        # Arguments
            in_grads: numpy array with shape (batch, time_steps, units*2), gradients to outputs
            inputs: numpy array with shape (batch, time_steps, in_features), same with forward inputs

        # Returns
            out_grads: numpy array with shape (batch, time_steps, in_features), gradients to inputs
        """
        mask = ~np.any(np.isnan(inputs), axis=2)
        in_grads_split = np.split(in_grads, 2, axis=2)
        forward_in_grads = in_grads_split[0]
        backward_in_grads = self._reverse_temporal_data(in_grads_split[1], mask)
        forward_out_grads = self.forward_rnn.backward(forward_in_grads, inputs)
        backward_out_grads = self.backward_rnn.backward(backward_in_grads, self._reverse_temporal_data(inputs, mask))
        return forward_out_grads + self._reverse_temporal_data(backward_out_grads, mask)

    def update(self, params):
        """Update parameters with new params
        """
        for k,v in params.items():
            if '/forward_kernel' in k:
                self.forward_rnn.kernel = v
            elif '/forward_recurrent_kernel' in k:
                self.forward_rnn.recurrent_kernel = v
            elif '/forward_bias' in k:
                self.forward_rnn.bias = v
            elif '/backward_kernel' in k:
                self.backward_rnn.kernel = v
            elif '/backward_recurrent_kernel' in k:
                self.backward_rnn.recurrent_kernel = v
            elif '/backward_bias' in k:
                self.backward_rnn.bias = v
        
    def get_params(self, prefix):
        """Return parameters and gradients
        
        # Arguments
            prefix: string, to contruct prefix of keys in the dictionary (usually is the layer-ith)

        # Returns
            params: dictionary, store parameters of this layer
            grads: dictionary, store gradients of this layer

            None: if not trainable
        """
        if self.trainable:
            params = {
                prefix+':'+self.name+'/forward_kernel': self.forward_rnn.kernel,
                prefix+':'+self.name+'/forward_recurrent_kernel': self.forward_rnn.recurrent_kernel,
                prefix+':'+self.name+'/forward_bias': self.forward_rnn.bias,
                prefix+':'+self.name+'/backward_kernel': self.backward_rnn.kernel,
                prefix+':'+self.name+'/backward_recurrent_kernel': self.backward_rnn.recurrent_kernel,
                prefix+':'+self.name+'/backward_bias': self.backward_rnn.bias
            }
            grads = {
                prefix+':'+self.name+'/forward_kernel': self.forward_rnn.kernel_grad,
                prefix+':'+self.name+'/forward_recurrent_kernel': self.forward_rnn.r_kernel_grad,
                prefix+':'+self.name+'/forward_bias': self.forward_rnn.b_grad,
                prefix+':'+self.name+'/backward_kernel': self.backward_rnn.kernel_grad,
                prefix+':'+self.name+'/backward_recurrent_kernel': self.backward_rnn.r_kernel_grad,
                prefix+':'+self.name+'/backward_bias': self.backward_rnn.b_grad
            }
            return params, grads
        else:
            return None