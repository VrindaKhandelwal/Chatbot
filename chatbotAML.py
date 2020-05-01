import numpy as np
import re
import json
from keras.layers import Input, Embedding, LSTM
from keras.layers import  GRU, Dense, Embedding
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.preprocessing.text import Tokenizer

def clean_text(text):
	'''Clean text by removing unnecessary characters and altering the format of words.'''
	text = text.lower()
	text = re.sub(r"i'm", "i am", text)
	text = re.sub(r"he's", "he is", text)
	text = re.sub(r"she's", "she is", text)
	text = re.sub(r"it's", "it is", text)
	text = re.sub(r"that's", "that is", text)
	text = re.sub(r"aren't", "are not", text)
	text = re.sub(r"what's", "that is", text)
	text = re.sub(r"where's", "where is", text)
	text = re.sub(r"how's", "how is", text)
	text = re.sub(r"\'ll", " will", text)
	text = re.sub(r"\'ve", " have", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"\'d", " would", text)
	text = re.sub(r"\'re", " are", text)
	text = re.sub(r"won't", "will not", text)
	text = re.sub(r"can't", "cannot", text)
	text = re.sub(r"n't", " not", text)
	text = re.sub(r"n'", "ng", text)
	text = re.sub(r"'bout", "about", text)
	text = re.sub(r"'til", "until", text)
	text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
	return text

#loading the data into two files

with open('Inputfile.txt', 'r') as filehandle:
	InputList= json.load(filehandle)
with open('Outputfile.txt', 'r') as filehandle:
	OutputList= json.load(filehandle)

print(type(InputList))
X=[]
Y=[]

#cleaning the data
for i in range(0,len(InputList)):
	begin="<BOS>"
	end = "<EOS>"
	#print(InputList[i])
	X.append(clean_text(InputList[i]))
	Y.append(begin+ clean_text(OutputList[i]) + end)



MAX_LEN_OF_SEQ=1000
EMBEDDING_DIM=50

embeddings_dict = {}
with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
	for line in f:
		values = line.split()
		word = values[0]
		vector = np.asarray(values[1:], "float32")
		embeddings_dict[word] = vector
f.close()

print(len(embeddings_dict))

with open('Combinedfile.txt', 'r') as filehandle:
    textlist= json.load(filehandle)
#print(textlist)
words = ' '.join(textlist).split()
MAX_NUM_WORDS=len(words)
print('Number of words in text file :', len(words))

#to create vocabulary
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(textlist)
sequences = tokenizer.texts_to_sequences(textlist)

dictionary = tokenizer.word_index
print(len(dictionary),len(tokenizer.word_counts))

num_words_in_vocab = min(MAX_NUM_WORDS, len(dictionary) + 1)
print(num_words_in_vocab)


embedding_matrix = np.zeros((num_words_in_vocab, EMBEDDING_DIM))
for word, i in dictionary.items():
    if i >= MAX_NUM_WORDS:
        continue
    embed_vector = embeddings_dict.get(word)
    if embed_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embed_vector
print("done")

print(len(X))

input_enc_sequences = tokenizer.texts_to_sequences(X)
input_dec_sequences = tokenizer.texts_to_sequences(Y)
print(len(input_enc_sequences))

output_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
output_tokenizer.fit_on_texts(Y)
output_integer_seq = output_tokenizer.texts_to_sequences(Y)

embed_outputs = output_tokenizer.word_index
print('Total unique words in the output: ' , len(embed_outputs))

num_words_output = len(embed_outputs) + 1


max_in_len = max(len(sen) for sen in input_enc_sequences)
print("Length of longest sentence in the output: ", max_in_len)

max_out_len = max(len(sen) for sen in input_dec_sequences)
print("Length of longest sentence in the output: %g", max_out_len)

encoder_input_data = pad_sequences(input_enc_sequences, maxlen=max_in_len, dtype='int32', padding='post', truncating='post')
decoder_input_data = pad_sequences(input_dec_sequences, maxlen=max_out_len, dtype='int32', padding='post', truncating='post')


print(type(encoder_input_data))




embedding_layer = Embedding(input_dim=num_words_in_vocab,
                            output_dim=EMBEDDING_DIM,  
                            weights=[embedding_matrix],
                            input_length=max_in_len)
print('Training model.')




decoder_output_data = np.zeros((len(X), max_out_len, num_words_output), dtype="float32")
print(decoder_output_data.shape)
for a, b in enumerate(decoder_input_data):
	for j, seq in enumerate(b):
		if( j>0):
			decoder_output_data[a][j][seq] = 1
print("the shape is :",decoder_output_data.shape)


LSTM_NODES=8

E_inputs_placeholder = Input(shape=(max_in_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(LSTM_NODES, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]
decoder_inputs_placeholder = Input(shape=(max_out_len,))

decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)


decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


model = Model([encoder_inputs_placeholder,
decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# from keras.utils import plot_model
# plot_model(model, to_file='model_plot4a.png', show_shapes=True, show_layer_names=True)

BATCH_SIZE=32
EPOCHS=15

r = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_output_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
)


# test_size1=0.2
# test_size2=0.3
# en_train, en_test, de_train, de_test = train_test_split(encoder_input_data, decoder_input_data, test_size=test_size1)
# en_train, en_val, de_train, de_val = train_test_split(en_train, de_train, test_size=test_size2)



