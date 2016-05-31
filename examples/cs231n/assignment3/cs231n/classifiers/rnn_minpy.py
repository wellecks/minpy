import minpy
import minpy.numpy as np
import numpy as py_np
import functools

from model import ModelBase
from cs231n.rnn_layers_minpy import *
from cs231n.layers_minpy import *

class CaptioningRNN(ModelBase):
  def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
               hidden_dim=128, cell_type='rnn', dtype=np.float32):
    if cell_type not in {'rnn', 'lstm'}:
      raise ValueError('Invalid cell_type "%s"' % cell_type)
    
    self.cell_type = cell_type
    self.dtype = dtype
    self.word_to_idx = word_to_idx
    self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
    self.params = {}

    self._null = word_to_idx['<NULL>']
    self._start = word_to_idx.get('<START>', None)
    self._end = word_to_idx.get('<END>', None)
    vocab_size = len(word_to_idx)
    
    # Initialize word vectors
    self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
    self.params['W_embed'] /= 100
    
    # Initialize CNN -> hidden state projection parameters
    self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
    self.params['W_proj'] /= np.sqrt(input_dim)
    self.params['b_proj'] = np.zeros(hidden_dim)

    # Initialize parameters for the RNN
    dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
    self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
    self.params['Wx'] /= np.sqrt(wordvec_dim)
    self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
    self.params['Wh'] /= np.sqrt(hidden_dim)
    self.params['b'] = np.zeros(dim_mul * hidden_dim)
    
    # Initialize output to vocab weights
    self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
    self.params['W_vocab'] /= np.sqrt(hidden_dim)
    self.params['b_vocab'] = np.zeros(vocab_size)
      
    # Cast parameters to correct dtype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)

  def loss_and_derivative(self, features, captions_in_dense, captions_out_dense, mask):
    def train_loss(features, captions_in_dense, captions_out_dense, mask, W_proj, W_embed, W_h, W_x, W_vocab, b_proj, b, b_vocab):
      N,D = features.shape
      _,T = captions.shape

      loss = 0.0
      h0, cache_imgproj = affine_forward(features, W_proj, b_proj)
    
      embed, cache_embed = word_embedding_forward(captions_in_dense, W_embed)
    
      if self.cell_type == 'rnn':
        rnn_out, cache_rnn = rnn_forward(embed, h0, W_x, W_h, b) 
      else:
        assert(False)

      affine_out, cache_affine = temporal_affine_forward(rnn_out, W_vocab, b_vocab) 
      
      loss, dsoftmax = temporal_softmax_loss(affine_out, captions_out_dense, mask) 
      
      return loss

    self.params_array = []
    params_list_name = ['W_proj', 'W_embed', 'W_h', 'W_x', 'W_vocab', 'b_proj', 'b', 'b_vocab']
    for param_name in params_list_name:
      self.params_array.append(self.params[param_name])

    grad_function = grad_and_loss(train_loss, range(2, 10))

    grads_array, loss = grad_function(features, captions, *self.params_array)

    grads = {}

    for i in range(len(params_list_name)):
      grads[params_list_name[i]] = grads_array[i]

    return loss, grads


