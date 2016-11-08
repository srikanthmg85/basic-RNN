#Toy example of seq2seq modeling using vanilla RNN 
# h(t) = tanh(Wxh*x(t) + Whh*h(t-1) + bh)
# y(t) = Wyh*h(t) + by 
# x(t) is n dimensional embedding for one hot vector for vocabulary of length m 

import numpy as np
import os
import sys
import pdb

class VanillaRNN:

  def __init__(self,txtFile,num_hidden_units,seq_len):
    f = open(txtFile,'r')
    self.data = f.read()
    chars = list(set(self.data))
    num_data = len(self.data)
    num_chars = len(chars) 
    self.char_to_index = {c:i for i,c in enumerate(chars)}
    self.index_to_char = {i:c for i,c in enumerate(chars)}
    self.num_hidden_units = num_hidden_units
    self.vocab_size = num_chars
    self.whh = np.random.randn(num_hidden_units,num_hidden_units)*0.01
    self.wxh = np.random.randn(num_hidden_units,self.vocab_size)*0.01
    self.wyh = np.random.randn(self.vocab_size,num_hidden_units)*0.01
    self.bh = np.zeros((num_hidden_units,1))
    self.by = np.zeros((self.vocab_size,1))
    self.seq_len = seq_len

  def forward(self,inputs,hprev):

    x = {}
    h1 = {}
    probs = {}

    h1[-1] = np.copy(hprev)
 
    for i in range(seq_len):
      x[i] = np.zeros((self.vocab_size,1))
      x[i][inputs[i]] = 1
      
      h1[i] = np.dot(self.wxh , x[i]) + np.dot(self.whh , h1[i-1]) + self.bh 
      h1[i] = np.tanh(h1[i])
      logits = np.dot(self.wyh, h1[i]) + self.by
      probs[i] = np.exp(logits)/sum(np.exp(logits))

    return x,h1,probs


  def loss(self,probs,targets):
    loss = 0
    for i in range(self.seq_len):
      loss += -np.log(probs[i][targets[i]])
    return loss



 
txtFile = sys.argv[1] 
num_hidden_units = 100
seq_len = 25 
rnn = VanillaRNN(txtFile,num_hidden_units,seq_len)


for i in range(len(rnn.data)-25):

  inputs = [rnn.char_to_index[ch] for ch in rnn.data[0:25]]
  outputs= [rnn.char_to_index[ch] for ch in rnn.data[1:26]]

  hprev = np.zeros((rnn.num_hidden_units,1))
  x,h1,probs = rnn.forward(inputs,hprev)

  loss = rnn.loss(probs,outputs)

  print loss








    
    
