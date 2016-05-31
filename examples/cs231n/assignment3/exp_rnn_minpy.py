from cs231n.rnn_layers_minpy import *

import minpy.numpy as minpy_np
import minpy.core
import minpy.array
from minpy.array_variants import ArrayType
import minpy.dispatch.policy as policy
from minpy.core import grad_and_loss

import numpy as np

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

def NumpyVarToMinpy(var):
  return minpy.array.Value.wrap(var)

def MinpyVarToNumpy(var):
  return minpy.array.Value.wrap(var).get_data(ArrayType.NUMPY)

def Word_Encode(x, W):
    out, cache = None, None
    # get shape
    N,T = x.shape
    V,D = W.shape
    x_sparse = np.zeros([N, T,V])
    for i in range(N):
        for j in range(T):
            x_sparse[i,j,x[i,j]] = 1
    return x_sparse

def Sparse_To_Dense(x, N, T, V):
    x_sparse = np.zeros([N, T,V])
    for i in range(N):
        for j in range(T):
            x_sparse[i,j,x[i,j]] = 1
    return x_sparse

def rel_error(x, y):
  a = np.max(np.abs(x - y) / (np.maximum(1e-5, np.abs(x) + np.abs(y))))
  if a > 0.5:
    print 'probably error'
    '''
    print np.abs(x - y) / (np.maximum(1e-5, np.abs(x) + np.abs(y)))
    print 'left'
    print x 
    print 'right'
    print y 
    '''
    maxidx = np.argmax(np.abs(x - y) / (np.maximum(1e-5, np.abs(x) + np.abs(y))))
    print x.flatten()[maxidx]
    print y.flatten()[maxidx]
  return a

# Forward step
def Test_Forward_Step():
  N, D, H = 3, 10, 4

  x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)
  prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)
  Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)
  Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)
  b = np.linspace(-0.2, 0.4, num=H)

  mp_x = NumpyVarToMinpy(x)
  mp_prev_h = NumpyVarToMinpy(prev_h)
  mp_Wx = NumpyVarToMinpy(Wx)
  mp_Wh = NumpyVarToMinpy(Wh)
  mp_b = NumpyVarToMinpy(b)
  
  mp_next_h, _ = rnn_step_forward(mp_x, mp_prev_h, mp_Wx, mp_Wh, mp_b)
  next_h = MinpyVarToNumpy(mp_next_h)
  
  expected_next_h = np.asarray([[-0.58172089, -0.50182032, -0.41232771, -0.31410098],[ 0.66854692,  0.79562378,  0.87755553,  0.92795967],[ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])

  print 'next_h error: ', rel_error(expected_next_h, next_h)

def Test_Backward_Step():
  N, D, H = 4, 5, 6
  x = np.random.randn(N, D)
  h = np.random.randn(N, H)
  Wx = np.random.randn(D, H)
  Wh = np.random.randn(H, H)
  b = np.random.randn(H)

  mp_x = NumpyVarToMinpy(x)
  mp_h = NumpyVarToMinpy(h)
  mp_Wx = NumpyVarToMinpy(Wx)
  mp_Wh = NumpyVarToMinpy(Wh)
  mp_b = NumpyVarToMinpy(b)

  mp_next_h, _ = rnn_step_forward(mp_x, mp_h, mp_Wx, mp_Wh, mp_b)
  out = MinpyVarToNumpy(mp_next_h)

  dnext_h = np.random.randn(*out.shape)
  mp_dnext_h = NumpyVarToMinpy(dnext_h)

  rnn_step_forward_with_loss = lambda mp_x, mp_h, mp_Wx, mp_Wh, mp_b, mp_dnext_h: rnn_step_forward(mp_x, mp_h, mp_Wx, mp_Wh, mp_b)[0] * mp_dnext_h  
 
  grad_function = grad_and_loss(rnn_step_forward_with_loss, range(0,5))

  grad_arrays, loss = grad_function(mp_x, mp_h, mp_Wx, mp_Wh, mp_b, mp_dnext_h)

  dx_num = eval_numerical_gradient_array(lambda x: MinpyVarToNumpy(rnn_step_forward(NumpyVarToMinpy(x), h, Wx, Wh, b)[0]), x, dnext_h)
  dh_num = eval_numerical_gradient_array(lambda h: MinpyVarToNumpy(rnn_step_forward(x, NumpyVarToMinpy(h), Wx, Wh, b)[0]), h, dnext_h)
  dWx_num = eval_numerical_gradient_array(lambda Wx: MinpyVarToNumpy(rnn_step_forward(x, h, NumpyVarToMinpy(Wx), Wh, b)[0]), Wx, dnext_h)
  dWh_num = eval_numerical_gradient_array(lambda Wh: MinpyVarToNumpy(rnn_step_forward(x, h, Wx, NumpyVarToMinpy(Wh), b)[0]), Wh, dnext_h)
  db_num = eval_numerical_gradient_array(lambda b: MinpyVarToNumpy(rnn_step_forward(x, h, Wx, Wh, NumpyVarToMinpy(b))[0]), b, dnext_h)

  
  print 'dx error: ', rel_error(dx_num, MinpyVarToNumpy(grad_arrays[0]))
  print 'dh error: ', rel_error(dh_num, MinpyVarToNumpy(grad_arrays[1]))
  print 'dWx error: ', rel_error(dWx_num, MinpyVarToNumpy(grad_arrays[2]))
  print 'dWh error: ', rel_error(dWh_num, MinpyVarToNumpy(grad_arrays[3]))
  print 'db error: ', rel_error(db_num, MinpyVarToNumpy(grad_arrays[4]))

def Test_RNN_Forward():
  N, T, D, H = 2, 3, 4, 5

  x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D) 
  h0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)
  Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)
  Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)
  b = np.linspace(-0.7, 0.1, num=H)

  mp_x = NumpyVarToMinpy(x)
  mp_h0 = NumpyVarToMinpy(h0)
  mp_Wx = NumpyVarToMinpy(Wx)
  mp_Wh = NumpyVarToMinpy(Wh)
  mp_b = NumpyVarToMinpy(b)

  h, _ = rnn_forward(mp_x, mp_h0, mp_Wx, mp_Wh, mp_b)
 
  expected_h = np.asarray([
      [
            [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],
                [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],
                    [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],
                      ],
        [
              [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],
                  [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],
                      [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])
  
  print 'h error: ', rel_error(expected_h, MinpyVarToNumpy(h))

def Test_RNN_Backward():
  N, D, T, H = 2, 3, 10, 5

  x = np.random.randn(N, T, D)
  h = np.random.randn(N, H)
  Wx = np.random.randn(D, H)
  Wh = np.random.randn(H, H)
  b = np.random.randn(H)

  mp_x = NumpyVarToMinpy(x)
  mp_h = NumpyVarToMinpy(h)
  mp_Wx = NumpyVarToMinpy(Wx)
  mp_Wh = NumpyVarToMinpy(Wh)
  mp_b = NumpyVarToMinpy(b)

  out, _ = rnn_forward(mp_x, mp_h, mp_Wx, mp_Wh, mp_b)

  dout = np.random.randn(*MinpyVarToNumpy(out).shape)
  mp_dout = NumpyVarToMinpy(dout)

  rnn_forward_with_loss = lambda mp_x, mp_h, mp_Wx, mp_Wh, mp_b, mp_out: rnn_forward(mp_x, mp_h, mp_Wx, mp_Wh, mp_b)[0] * mp_out 
  
  grad_function = grad_and_loss(rnn_forward_with_loss, range(0,5))

  grad_arrays, loss = grad_function(mp_x, mp_h, mp_Wx, mp_Wh, mp_b, mp_dout)

  dx_num = eval_numerical_gradient_array(lambda x: MinpyVarToNumpy(rnn_forward(NumpyVarToMinpy(x), h, Wx, Wh, b)[0]), x, dout)
  dh_num = eval_numerical_gradient_array(lambda h: MinpyVarToNumpy(rnn_forward(x, NumpyVarToMinpy(h), Wx, Wh, b)[0]), h, dout)
  dWx_num = eval_numerical_gradient_array(lambda Wx: MinpyVarToNumpy(rnn_forward(x, h, NumpyVarToMinpy(Wx), Wh, b)[0]), Wx, dout)
  dWh_num = eval_numerical_gradient_array(lambda Wh: MinpyVarToNumpy(rnn_forward(x, h, Wx, NumpyVarToMinpy(Wh), b)[0]), Wh, dout)
  db_num = eval_numerical_gradient_array(lambda b: MinpyVarToNumpy(rnn_forward(x, h, Wx, Wh, NumpyVarToMinpy(b))[0]), b, dout)
  
  print 'dx error: ', rel_error(dx_num, MinpyVarToNumpy(grad_arrays[0]))
  print 'dh error: ', rel_error(dh_num, MinpyVarToNumpy(grad_arrays[1]))
  print 'dWx error: ', rel_error(dWx_num, MinpyVarToNumpy(grad_arrays[2]))
  print 'dWh error: ', rel_error(dWh_num, MinpyVarToNumpy(grad_arrays[3]))
  print 'db error: ', rel_error(db_num, MinpyVarToNumpy(grad_arrays[4]))

# embed forward
def Test_Embed_Forward():
    N, T, V, D = 2, 4, 5, 3

    x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])
    W = np.linspace(0, 1, num=V*D).reshape(V, D)
    x_sparse = Word_Encode(x, W)

    mp_x_sparse = NumpyVarToMinpy(x_sparse)
    mp_W = NumpyVarToMinpy(W)
    
    out, _ = word_embedding_forward(mp_x_sparse, mp_W)
    
    expected_out = np.asarray([
       [[ 0.,          0.07142857,  0.14285714],
           [ 0.64285714,  0.71428571,  0.78571429],
             [ 0.21428571,  0.28571429,  0.35714286],
               [ 0.42857143,  0.5,         0.57142857]],
        [[ 0.42857143,  0.5,         0.57142857],
            [ 0.21428571,  0.28571429,  0.35714286],
              [ 0.,          0.07142857,  0.14285714],
                [ 0.64285714,  0.71428571,  0.78571429]]])
    print 'out error: ', rel_error(expected_out, MinpyVarToNumpy(out))

# embed backward
def Test_Embed_Backward():
    N, T, V, D = 50, 3, 5, 6

    x = np.random.randint(V, size=(N, T))
    W = np.random.randn(V, D)
    x_sparse = Word_Encode(x, W)

    mp_x_sparse = NumpyVarToMinpy(x_sparse)
    mp_W = NumpyVarToMinpy(W)
    out, _ = word_embedding_forward(mp_x_sparse, mp_W)
    
    dout = np.random.randn(*out.shape)
    mp_dout = NumpyVarToMinpy(dout)

    embed_forward_with_loss = lambda mp_x_sparse, mp_W, mp_out: word_embedding_forward(mp_x_sparse, mp_W)[0] * mp_out
  
    grad_function = grad_and_loss(embed_forward_with_loss, range(0,2))

    grad_arrays, loss = grad_function(mp_x_sparse, mp_W, mp_dout)
    dW = MinpyVarToNumpy(grad_arrays[1])

    dW_num = eval_numerical_gradient_array(lambda W: MinpyVarToNumpy(word_embedding_forward(mp_x_sparse, NumpyVarToMinpy(W))[0]), W, dout)

    print 'dW error: ', rel_error(dW, dW_num)

# Caption
def Text_Caption():
    N, D, W, H = 10, 20, 30, 40
    word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}
    V = len(word_to_idx)
    T = 13

    model = CaptioningRNN(word_to_idx,
                  input_dim=D,
                            wordvec_dim=W,
                                      hidden_dim=H,
                                                cell_type='rnn',
                                                          dtype=np.float64)

    # Set all model parameters to fixed values
    for k, v in model.params.iteritems():
        model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)

    features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)
    captions = (np.arange(N * T) % V).reshape(N, T)
    
    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    mask = (captions_out != self._null)

    captions_in_dense = Sparse_To_Dense(captions_in, N,T-1,V)
    captions_out_dense = Sparse_To_Dense(captions_in, N,T-1,V)
    
    loss, grads = model.loss(features, captions_in_dense, captions_out_dense, mask)
    expected_loss = 9.83235591003

    print 'loss: ', loss
    print 'expected loss: ', expected_loss
    print 'difference: ', abs(loss - expected_loss)

#Test_Forward_Step()
#Test_Backward_Step()
#Test_RNN_Forward()
#Test_RNN_Backward()
#Test_Embed_Forward()
#Test_Embed_Backward()
Test_Caption()

