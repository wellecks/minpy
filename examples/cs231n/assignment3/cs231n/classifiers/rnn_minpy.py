import minpy.numpy as np
import numpy as nnp

from cs231n.rnn_layers_minpy import *
from cs231n.layers_minpy import *

def NumpyVarToMinpy(var):
  return minpy.array.Value.wrap(var)

def MinpyVarToNumpy(var):
  return minpy.array.Value.wrap(var).get_data(ArrayType.NUMPY)

def Word_Encode(x, W):
    out, cache = None, None
    # get shape
    N,T = x.shape
    V,D = W.shape
    x_sparse = nnp.zeros([N, T,V])
    for i in range(N):
        for j in range(T):
            x_sparse[i,j,x[i,j]] = 1
    return x_sparse

class CaptioningRNN(object):
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

  def loss(self, features, captions):
    N,D = features.shape
    _,T = captions.shape

    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]
    mask = (captions_out != self._null)

    # Weight and bias for the affine transform from image features to initial
    # hidden state
    W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
    
    # Word embedding matrix
    W_embed = self.params['W_embed']

    # Input-to-hidden, hidden-to-hidden, and biases for the RNN
    Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

    # Weight and bias for the hidden-to-vocab transformation.
    W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']
    
    loss, grads = 0.0, {}
    
    # (1) Use an affine transformation to compute the initial hidden state     #
    #     from the image features. This should produce an array of shape (N, H)#
    h0, cache_imgproj = affine_forward(features, W_proj, b_proj)
    
    # (2) Use a word embedding layer to transform the words in captions_in     #
    #     from indices to vectors, giving an array of shape (N, T, W).         #
    caption_in_sparse = NumpyVarToMinpy(Word_Encode(captions_in, MinpyVarToNumpy(W_embed)))
    embed, cache_embed = word_embedding_forward(captions_in_sparse, W_embed)
    
    # (3) Use either a vanilla RNN or LSTM (depending on self.cell_type) to    #
    #     process the sequence of input word vectors and produce hidden state  #
    #     vectors for all timesteps, producing an array of shape (N, T, H).    #
    if self.cell_type == 'rnn':
      rnn_out, cache_rnn = rnn_forward(embed, h0, Wx, Wh, b) 
    else:
      assert(False)

    # (4) Use a (temporal) affine transformation to compute scores over the    #
    #     vocabulary at every timestep using the hidden states, giving an      #
    #     array of shape (N, T, V).                                            #
    affine_out, cache_affine = temporal_affine_forward(rnn_out, W_vocab, b_vocab) 
    
    # (5) Use (temporal) softmax to compute loss using captions_out, ignoring  #
    #     the points where the output word is <NULL> using the mask above.     #
    loss, dsoftmax = temporal_softmax_loss(affine_out, captions_out, mask) 
    
    grads = None
    return loss, grads









