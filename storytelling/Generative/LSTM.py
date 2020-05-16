#import lib
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences 
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional 
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from attention import AttentionLayer
import os
pd.set_option("display.max_colwidth", 200) 

warnings.filterwarnings("ignore")
#%%LOad Data
# Remark : Encoding UTF-8 has been fixed in helper.py
def load_data(path):
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read()

    return data.split('\n')


event_tr = load_data('data2/genEventsToGenSentsTrain_input.txt')
event_val= load_data('data2/genEventsToGenSentsValidation_input.txt')

sen_tr = load_data('data2/genEventsToGenSentsTrain_output.txt')
sen_val=load_data('data2/genEventsToGenSentsValidation_output.txt')
sen_tr=list(map(lambda x : x[:-2],sen_tr))
sen_val=list(map(lambda x : x[:-2],sen_val))

max_len_event=4
max_len_sen=30

short_event_tr=[]
short_sen_tr=[]
for i in range(0,len(sen_tr)):
    if len(sen_tr[i].split())<=max_len_sen:
        short_event_tr.append(event_tr[i])
        short_sen_tr.append(sen_tr[i])
        
short_event_val=[]
short_sen_val=[]
for i in range(0,len(sen_val)):
    if len(sen_val[i].split())<=max_len_sen:
        short_event_val.append(event_val[i])
        short_sen_val.append(sen_val[i])

short_sen_tr=list(map(lambda x : '_START_ '+ x + ' _END_',short_sen_tr))
short_sen_val=list(map(lambda x : '_START_ '+ x + ' _END_',short_sen_val))

#%%
#INPUT_tokenization

x_tokenizer = Tokenizer(filters='!"#$%&*+,/:;=?@[\\]^`{|}~\t\n',split=' ') #prepare a tokenizer for events on training data 
x_tokenizer.fit_on_texts(short_event_tr)
x_tr = x_tokenizer.texts_to_sequences(short_event_tr)#convert text sequences into integer sequences
x_val = x_tokenizer.texts_to_sequences(short_event_val)
x_tr = pad_sequences(x_tr, maxlen=max_len_event, padding='post')#padding zero upto maximum length
x_val = pad_sequences(x_val, maxlen=max_len_event, padding='post')
x_voc_size = len(x_tokenizer.word_index) +1

#OUTPUT_tokenization


y_tokenizer = Tokenizer(filters='!"#$%&*+,/:;=?@[\\]^`{|}~\t\n',split=' ') 
y_tokenizer.fit_on_texts(short_sen_tr)
#convert summary sequences into integer sequences 
y_tr = y_tokenizer.texts_to_sequences(short_sen_tr)
y_val = y_tokenizer.texts_to_sequences(short_sen_val)
#padding zero upto maximum length
y_tr = pad_sequences(y_tr, maxlen=max_len_sen, padding='post') 
y_val = pad_sequences(y_val, maxlen=max_len_sen, padding='post')
y_voc_size = len(y_tokenizer.word_index) +1

#%%
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
tf.compat.v1.keras.backend.clear_session()  # For easy reset of notebook state.

config_proto = tf.compat.v1.ConfigProto()
off = rewriter_config_pb2.RewriterConfig.OFF
config_proto.graph_options.rewrite_options.arithmetic_optimization = off
session = tf.compat.v1.Session(config=config_proto)
set_session(session)
#from keras import backend as K 
#K.clear_session()
emb_dim = 128
latent_dim=200

#Input & embed

encoder_inputs=Input(shape=(max_len_event,),name='enc_input')
enc_emb=Embedding(x_voc_size,emb_dim,trainable=True,name='enc_embedding')(encoder_inputs)

#LSTM

encoder_lstm=LSTM(latent_dim, return_state=True, return_sequences=True,name='enc_lstm2') 
encoder_outputs, state_h, state_c= encoder_lstm(enc_emb)

#Input & Embedding

decoder_inputs = Input(shape=(None,),name='dec_input')
dec_emb_layer=Embedding(y_voc_size, emb_dim,trainable=True,name='dec_embedding') 
dec_emb = dec_emb_layer(decoder_inputs)


#LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='dec_lstm')
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h, state_c])

# Attention layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])
# Concat attention input and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# Dense Layer
decoder_dense=TimeDistributed(Dense(y_voc_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model2 = Model(inputs=[encoder_inputs, decoder_inputs],outputs=decoder_outputs)
#%%
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy') #loss 也可以是mse

#callback
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2) 
#%%
#训练模型
history=model.fit([x_tr,y_tr[:,:-1]], 
                  y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:],
                  epochs=6,
                  callbacks=[es],
                  batch_size=64, 
                  validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))

#%% Lemma 调小xy的size
x_tr=x_tr[:10000]
y_tr=y_tr[:10000]
x_val=x_tr[:1000]
y_val=y_val[:1000]




#%%Decoder Inference
reverse_target_word_index=y_tokenizer.index_word
reverse_source_word_index=x_tokenizer.index_word
target_word_index=y_tokenizer.word_index

# Encode the input sequence to get the feature vector
encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

# Decoder setup
# Below tensors will hold the states of the previous time step 
decoder_state_input_h = Input(shape=(latent_dim,)) 
decoder_state_input_c = Input(shape=(latent_dim,)) 
decoder_hidden_state_input = Input(shape=(max_len_event,latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2= dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

#attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])
# A dense softmax layer to generate prob dist. over the target vocabulary 
decoder_outputs2 = decoder_dense(decoder_inf_concat)

decoder_model = Model(
[decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],\
 [decoder_outputs2] + [state_h2, state_c2])
#%%
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Chose the 'start' word as the first word of the target sequence 
    target_seq[0, 0] = target_word_index['_start_']
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]
        if(sampled_token!='_end_'):
            decoded_sentence += ' '+sampled_token
        
        # Exit condition: either hit max length or find stop word.
        if (sampled_token == '_end_' or len(decoded_sentence.split()) >= (max_len_sen-1)):
                stop_condition = True
        # Update the target sequence (of length 1). 
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
        # Update internal states
        e_h, e_c = h, c
    return decoded_sentence
#%%
import h5py
model2.load_weights('model_weight.h5')
#%%
from tensorflow.keras.models import load_model

model2=load_model('LSTM.h5',custom_objects={'AttentionLayer': AttentionLayer})
#%%
# Add-on : Check framework versions being used 
import tensorflow as tf
import keras as K
import sys
print("Python", sys.version)
print("Tensorflow version", tf.__version__)
print("Keras version", K.__version__)

#%%
decode_sequence(x_tr[191].reshape(1,max_len_event))[0]
