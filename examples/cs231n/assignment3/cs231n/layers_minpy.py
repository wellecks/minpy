import minpy.numpy as np

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
  out = np.reshape(x, [x.shape[0], np.prod(x.shape[1:])]).dot(w) + b
  cache = (x, w, b)
  return out, cache

def relu_forward(x):
  out = np.maximum(0, x)
  cache = x
  return out, cache




