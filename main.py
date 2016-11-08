#Toy example of seq2seq modeling using vanilla RNN 
# h(t) = tanh(Wxh*x(t) + Whh*h(t-1) + bh)
# y(t) = Wyh*h(t) + by 
# x(t) is n dimensional embedding for one hot vector for vocabulary of length m 

import numpy as np
import os
import sys
import pdb

class VanillaRNN:

  def __init__(self,txtFile,num_hidden_units,seq_len,learning_rate,num_epochs):
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
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs 

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

    return x,h1,probs,h1[seq_len-1]


  def loss(self,probs,targets):
    loss = 0
    for i in range(self.seq_len):
      loss += -np.log(probs[i][targets[i]])
    return loss


  def backward(self,probs,targets,x,h1):
    dh_next = np.zeros_like(h1[0])
    dyh = np.zeros_like(self.wyh)
    dxh = np.zeros_like(self.wxh)
    dhh = np.zeros_like(self.whh)
    dbh = np.zeros_like(self.bh)
    dby = np.zeros_like(self.by)
    dh = np.zeros_like(h1[0])

    for i in reversed(range(len(targets))):
      dy = np.copy(probs[i])
      dy[targets[i]] -= 1    

      #backpropogate to wyh 
      dyh += np.dot(dy,h1[i].T)
      dby += dy     

      #backpropogate to h 
      dh = np.dot(self.wyh.T,dy) + dh_next
      

      #backpropogate to h before non-linearity
      dh_bef = dh*(1-h1[i]*h1[i])      
      dbh += np.sum(dh_bef)

      #backpropogate to wxh
      dxh += np.dot(dh_bef,x[i].T)

      #backpropogate to whh 
      dhh += np.dot(dh_bef,h1[i-1].T)

      dh_next = np.dot(self.whh.T,dh_bef)

      return dxh,dhh,dyh,dbh,dby

  def SGD_step(self,dxh,dhh,dyh,dbh,dby):
    
    np.clip(dxh,-5,5,out=dxh)
    np.clip(dhh,-5,5,out=dhh)
    np.clip(dyh,-5,5,out=dyh)
    self.wxh -= self.learning_rate*dxh
    self.whh -= self.learning_rate*dhh
    self.wyh -= self.learning_rate*dyh
    self.bh  -= self.learning_rate*dbh
    self.by  -= self.learning_rate*dby

 
txtFile = sys.argv[1] 
num_hidden_units = 100
seq_len = 25 
rnn = VanillaRNN(txtFile,num_hidden_units,seq_len,0.1,10)

hprev = np.zeros((rnn.num_hidden_units,1))


for j in range(rnn.num_epochs):
  loss = 0
  for i in range(len(rnn.data)/rnn.seq_len):

    inputs = [rnn.char_to_index[ch] for ch in rnn.data[i*rnn.seq_len:(i+1)*rnn.seq_len]]
    outputs= [rnn.char_to_index[ch] for ch in rnn.data[i*rnn.seq_len + 1:(i+1)*rnn.seq_len + 1]]

    x,h1,probs,hprev = rnn.forward(inputs,hprev)
    loss = rnn.loss(probs,outputs)
    

    dwxh,dwhh,dwyh,dbh,dby = rnn.backward(probs,outputs,x,h1)
    rnn.SGD_step(dwxh,dwhh,dwyh,dbh,dby)
 
     








    
    
