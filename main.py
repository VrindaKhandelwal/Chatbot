#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.cost import cross_entropy_seq, cross_entropy_seq_with_mask
from tqdm import tqdm
from sklearn.utils import shuffle
from data import data
from tensorlayer.models.seq2seq import Seq2seq
from tensorlayer.models.seq2seq_with_attention import Seq2seqLuongAttention
import os

main_path = 'data/'
model_file = f"{main_path}model.npz"
data_corpus = "adhiraj_chat" # which data set
decoder_seq_length = 20 # max words in the decoded sequence

test_data = ["love you", 
             "happy birthday bro", 
             "can you come to the mess", 
             "have you fixed the issue",
             "good morning"]

def initial_setup(data_corpus):
    '''
      Load all the data including splitting into train & test data
    '''
    metadata, idx_q, idx_a = load_data(PATH=f"{main_path}{data_corpus}/")
    (trainX, trainY), (testX, testY), (_, _) = split_dataset(idx_q, idx_a)
    trainX = tl.prepro.remove_pad_sequences(trainX.tolist())
    trainY = tl.prepro.remove_pad_sequences(trainY.tolist())
    testX = tl.prepro.remove_pad_sequences(testX.tolist())
    testY = tl.prepro.remove_pad_sequences(testY.tolist())
    return metadata, trainX, trainY, testX, testY

def inference(seed, k):
  '''
    Given a query, predict the k-th best response
  '''
  model_.eval() # set model in evaluation mode (weigthts are not modified)
  seed_id = [word2idx.get(w, unk_id) for w in seed.split(" ")] # convert sentence to sequence of IDs
  sentence_id = model_(inputs=[[seed_id]], start_token=start_id, top_n=k) # predict
  
  sentence = [] 
  unknowns = 0
  for w_id in sentence_id[0]:
      w = idx2word[w_id]
      if w_id == unk_id:
          unknowns += 1
          w = ''
      if w == 'end_id':
          break
      sentence.append(w)
  return ' '.join (sentence), unknowns
def chat ():
  '''
    Small chat program
  '''
  print ("type 'quit' to exit")
  while True:
    query = input ("You: ")
    if query == "q" or query == "quit":
      break
    top_n = 3
    best_sentence, low_unknowns = None, None
    for _ in range(top_n):
        sentence, unknowns = inference(query, top_n)
        print(f" possible output: {sentence} ({unknowns} unknowns)")
        if low_unknowns is None or unknowns < low_unknowns:
          best_sentence, low_unknowns = sentence, unknowns
        if unknowns == 0:
          break
    print (f"Bot: {best_sentence}")
def train_model ():
  optimizer = tf.optimizers.Adam(learning_rate=0.001)

  for epoch in range(num_epochs):

      for seed in seeds: # make some predictions before training this epoch
          print("Q >", seed)
          for i in range(3):
              sentence, unknowns = inference(seed, 3)
              print(f"> {sentence} ({unknowns} unknowns)")

      model_.train() # puts the model in training mode
      #trainX, trainY = shuffle(trainX, trainY, random_state=0) # do not shuffle the training data
      # iterate over the data in batches
      total_loss, n_iter = 0, 0
      for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                      total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

          X = tl.prepro.pad_sequences(X)
          _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
          _target_seqs = tl.prepro.pad_sequences(_target_seqs, maxlen=decoder_seq_length)
          _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
          _decode_seqs = tl.prepro.pad_sequences(_decode_seqs, maxlen=decoder_seq_length)
          _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

          with tf.GradientTape() as tape:
              ## compute outputs
              output = model_(inputs = [X, _decode_seqs])
              output = tf.reshape(output, [-1, vocabulary_size])
              # compute loss
              loss = cross_entropy_seq_with_mask(logits=output, target_seqs=_target_seqs, input_mask=_target_mask)
              # apply the gradients
              grad = tape.gradient(loss, model_.all_weights)
              optimizer.apply_gradients(zip(grad, model_.all_weights))
          total_loss += loss
          n_iter += 1
      # printing average loss after every epoch
      print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))
      tl.files.save_npz(model_.all_weights, name=model_file) # save the weights to the file after every epoch


#data preprocessing
metadata, trainX, trainY, testX, testY = initial_setup(data_corpus)

word2idx = metadata['w2idx'] # dict  word 2 index
idx2word = metadata['idx2w'] # list index 2 word

src_len = len (trainX)

batch_size = 192
n_step = src_len // batch_size
emb_dim = 1024 # embedding dictionary size
num_epochs = 20 # epochs to train for

vocabulary_size = len(word2idx)

start_id = word2idx['start_id'] # <bos>
end_id = word2idx['end_id'] # <eos>

unk_id = word2idx['unk'] # id for an unknown word
pad_id = word2idx['_'] # id for padding

# construct the model
embedding_layer = tl.layers.Embedding(vocabulary_size=vocabulary_size, embedding_size=emb_dim)
model_ = Seq2seq(decoder_seq_length=decoder_seq_length,
                  cell_enc=tf.keras.layers.GRUCell,
                  cell_dec=tf.keras.layers.GRUCell,
                  n_layer=3, # number of GRU layers
                  n_units=300, 
                  embedding_layer=embedding_layer)

# load the model if its there
if os.path.isfile (model_file):
    load_weights = tl.files.load_npz(name=model_file)
    tl.files.assign_weights(load_weights, model_)
    print ("loaded weights")

chat () # chat with the model
train_model () # train it also