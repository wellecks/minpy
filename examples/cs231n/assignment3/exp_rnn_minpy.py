from cs231n.rnn_layers_minpy import *

import numpy as np

import minpy 
import minpy.numpy as minpy_np
import minpy.core
import minpy.array
from minpy.array_variants import ArrayType
import minpy.dispatch.policy as policy
from minpy.core import grad_and_loss

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array

def NumpyVarToMinpy(var):
  return minpy.array.Value.wrap(var)

def MinpyVarToNumpy(var):
  return minpy.array.Value.wrap(var).get_data(ArrayType.NUMPY)

def rel_error(x, y):
  a = np.max(np.abs(x - y) / (np.maximum(1e-5, np.abs(x) + np.abs(y))))
  if a > 0.5:
    print 'probably error'
    print np.abs(x - y) / (np.maximum(1e-5, np.abs(x) + np.abs(y)))
    print 'left'
    print x 
    print 'right'
    print y 
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
  
  print 'h error: ', rel_error(expected_h, h)

def Test_RNN_Backward():
  N, D, T, H = 2, 3, 1, 5

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
  '''
  print 'dh error: ', rel_error(dh_num, MinpyVarToNumpy(grad_arrays[1]))
  print 'dWx error: ', rel_error(dWx_num, MinpyVarToNumpy(grad_arrays[2]))
  print 'dWh error: ', rel_error(dWh_num, MinpyVarToNumpy(grad_arrays[3]))
  print 'db error: ', rel_error(db_num, MinpyVarToNumpy(grad_arrays[4]))
  '''

#Test_Forward_Step()
#Test_Backward_Step()
#Test_RNN_Forward()
Test_RNN_Backward()
