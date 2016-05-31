import minpy.numpy as np
import numpy as py_np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  # TODO here has to use py_np
  out = np.dot(np.reshape(x, [x.shape[0], py_np.prod(x.shape[1:])]), w)
  out = out + b
  cache = (x, w, b)
  return out, cache

def relu_forward(x):
  out = np.maximum(0, x)
  cache = x
  return out, cache




