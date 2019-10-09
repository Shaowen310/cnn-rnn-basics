import numpy as np

def get_img2col_indices(x_shape, kernel_height, kernel_width, padding, stride):
    N, C, H, W = x_shape

    out_height = int((H + 2 * padding - kernel_height) / stride + 1)
    out_width = int((W + 2 * padding - kernel_width) / stride + 1)

    i0 = np.repeat(np.arange(kernel_height), kernel_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(kernel_width), kernel_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), kernel_height * kernel_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)

def img2col_indices(x, kernel_height, kernel_width, padding, stride):
    x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')

    k, i, j = get_img2col_indices(x.shape, kernel_height, kernel_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kernel_height * kernel_width * C, -1)
    return cols

def col2img_indices(cols, x_shape, kernel_height, kernel_width, padding,
                    stride):
    N, C, H, W = x_shape

    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)

    k, i, j = get_img2col_indices(x_shape, kernel_height, kernel_width, padding, stride)
    cols_reshaped = cols.reshape(C * kernel_height * kernel_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def maxpool(X_col):
    max_idx = np.argmax(X_col, axis=0)
    out = X_col[max_idx, range(max_idx.size)]
    return out, max_idx

def dmaxpool(dX_col, dout_col, max_idx):
    dX_col[max_idx, range(dout_col.size)] = dout_col
    return dX_col

def avgpool(X_col):
    out = np.mean(X_col, axis=0)
    return out, None

def davgpool(dX_col, dout_col, dummy_arg):
    dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
    return dX_col
