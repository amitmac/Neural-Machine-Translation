import gensim
import numpy as np
import tensorflow as tf
import re
import nltk

import data_utils as utils
from attention_models import Bahadanau

src_vocab_path = 'data/src-vocab.txt'
targ_vocab_path = 'data/targ-vocab.txt'

# Using whole vocabulary
# Vocabulary size for source data
with tf.gfile.GFile(src_vocab_path, mode="rb") as src_vocab_file:
    src_vocabulary_size = len(src_vocab_file.readlines())

# Vocabulary size for target data
with tf.gfile.GFile(targ_vocab_path, mode="rb") as targ_vocab_file:
    targ_vocabulary_size = len(targ_vocab_file.readlines())

embedding_size = 300

# Create embedding lookup
# This stores the embeddings for all the words in vocabulary so when a list of ids is passed
# in the input, those ids will be looked up in the src_embeddings to get the actual input
src_embeddings = tf.placeholder(shape=(src_vocabulary_size, embedding_size), 
                                dtype=tf.float32, 
                                name='src_embeddings')

# Create placeholder for encoder input
# Input dimension would be (batch_size x length_of_sentence)
#   - batch_size could vary so kept None
#   - length of input sentence could vary for which we have created buckets of (5, 10, 20, 40)
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')

# Lookup the encoder input ids in the src_embeddings
encoder_inputs_embedded = tf.nn.embedding_lookup(src_embeddings, encoder_inputs)

# Define the number of neurons in the rnn cell
encoder_hidden_units = 300

# Define the RNN encoder cell
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

# Create the RNN by specifying above defined cell
# time_major=False denotes that encoder_inputs and encoder_outputs is of size
# (batch_size, max_time_steps, ...), encoder_final_state is a tuple (c, h)
(encoder_outputs, encoder_final_state) = tf.nn.dynamic_rnn(encoder_cell, 
                                                           encoder_inputs_embedded, 
                                                           dtype=tf.float32, 
                                                           time_major=False)

### Decoder ###

# Create embedding lookup for target/decoder
# This stores the embeddings for all the words in vocabulary so when a list of ids is passed 
# in the input, those ids will be looked up in the src_embeddings to get the actual input
targ_embeddings = tf.placeholder(shape=(targ_vocabulary_size, embedding_size), 
                                 dtype=tf.float32, 
                                 name='targ_embeddings')

# Create placeholder for decoder target
# Target dimension would be (batch_size x length_of_sentence)
#   - length of input sentence could vary for which we have created buckets of (10, 15, 25, 50)
batch_size, _ = encoder_inputs.get_shape().as_list() # get batch size
decoder_targets = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_targets')

# Define the decoder_targets sequence length
decoder_sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32, 
                                         name='decoder_sequence_length')

# Lookup the encoder input ids in the targ_embeddings
decoder_targets_embedded = tf.nn.embedding_lookup(targ_embeddings, decoder_targets)

# Number of hidden units same as encoder hidden units as the encoder's final state is passed 
# to the decoder state initially
decoder_hidden_units = encoder_hidden_units

# Define the decoder cell
decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units, reuse=True)


# Padding
pad = utils.PAD_ID*tf.ones([tf.shape(encoder_inputs)[0]],dtype=tf.int32)
eos = utils.EOS_ID*tf.ones([tf.shape(encoder_inputs)[0]],dtype=tf.int32)
pad_embedded = tf.nn.embedding_lookup(targ_embeddings, pad)
eos_embedded = tf.nn.embedding_lookup(targ_embeddings, eos)

# Define the weights and biases for calculating the output word for the decoder cell 
# which will be fed into into next time step decoder cell
W_out = tf.Variable(tf.random_uniform([decoder_hidden_units, targ_vocabulary_size],-1,1), 
                    dtype=tf.float32)
b_out = tf.Variable(tf.zeros([targ_vocabulary_size]), dtype=tf.float32)

use_attention = True
if use_attention:
    attn = Bahadanau(decoder_hidden_units)

# We use raw_rnn to define the decoder as here we can control the input that goes into the
# next time step. raw_rnn takes the decoder_cell and a callback function as the input arguments.
# This callback function takes time, previous_output, previous_state and previous_loop_state  
# as the input and returns the next input, next cell output, next cell state, next loop state
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_sequence_length)
    initial_input = eos_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None
    
    return (initial_elements_finished, initial_input, 
            initial_cell_state, initial_cell_output, 
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    
    elements_finished = (time >= decoder_sequence_length)
    finished = tf.reduce_all(elements_finished)
    
    def get_next_input(): 
        if use_attention:
            h_bar = attn.bahadanau_model_single_step(previous_output,encoder_outputs,decoder_hidden_units)
        else:
            h_bar = previous_output
        
        # calculate the outputs
        output_logits = tf.add(tf.matmul(h_bar, W_out),b_out)
        output_word_index = tf.argmax(output_logits, 1)
        next_input = tf.nn.embedding_lookup(targ_embeddings, output_word_index)
        
        return next_input
    
    next_input = tf.cond(finished, lambda: pad_embedded, get_next_input)
    
    state = previous_state
    output = previous_output
    loop_state = None
    
    return (elements_finished, next_input, state, output, loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

# raw_rnn: decoder_outputs_ta - (batch_size, max_time_steps, hidden_size)
# decoder_final_states - (c, h): (batch_size, hidden_size)
decoder_outputs_ta, decoder_final_states, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

# pass all the outputs through attention model to get final outputs
decoder_attn_outputs = attn.bahadanau_model_multi_step(decoder_outputs, encoder_outputs)
 
_ , decoder_max_time_steps, _ = decoder_attn_outputs.get_shape().as_list()

# calculate logits 
decoder_outputs_flat = tf.reshape(decoder_attn_outputs, [-1, decoder_dim])
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W_out), b_out)
decoder_logits = tf.reshape(decoder_logits_flat, 
                            [-1, decoder_max_time_steps, targ_vocabulary_size])

# final prediction
decoder_prediction = tf.argmax(decoder_logits, 2)

# cross entropy loss
labels = tf.one_hot(decoder_targets,depth=targ_vocabulary_size, dtype=tf.float32)
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                 logits=decoder_logits)
# loss function
loss = tf.reduce_mean(stepwise_cross_entropy)

# train optimizer
train_op = tf.train.AdamOptimizer().minimize(loss)

# initialize all the variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

##################### Read data #####################

buckets = [(7, 13), (15, 20), (25, 30), (35, 40), (45, 50), (55,60)]

def read_data(src_path, targ_path):
    data_set = [{'src':[],'targ':[]} for _ in buckets]
    with tf.gfile.GFile(src_path, mode="rb") as src_file:
        with tf.gfile.GFile(targ_path, mode="rb") as targ_file:
            src, targ = src_file.readline(), targ_file.readline()
            while src and targ:
                src_ids = [int(x) for x in src.split()]
                targ_ids = [int(x) for x in targ.split()]
                
                targ_ids.append(utils.PAD_ID)
                
                for bucket_id, (src_size, targ_size) in enumerate(buckets):
                    if len(src_ids) <= src_size and len(targ_ids) <= targ_size:
                        src_ids += [utils.PAD_ID]*(src_size - len(src_ids))
                        targ_ids += [utils.PAD_ID]*(targ_size - len(targ_ids))
                        
                        data_set[bucket_id]['src'].append(src_ids)
                        data_set[bucket_id]['targ'].append(targ_ids)
                        
                        break
                
                src, targ = src_file.readline(), targ_file.readline()
    return data_set

# tokenized data path
src_train_tokenized_path = "data/tokenized/src-train-tokenized.txt"
targ_train_tokenized_path = "data/tokenized/targ-train-tokenized.txt"
src_val_tokenized_path = "data/tokenized/src-val-tokenized.txt"
targ_val_tokenized_path = "data/tokenized/targ-val-tokenized.txt"
src_test_tokenized_path = "data/tokenized/src-test-tokenized.txt"
targ_test_tokenized_path = "data/tokenized/targ-test-tokenized.txt"

train_set = read_data(src_train_tokenized_path, targ_train_tokenized_path)
val_set = read_data(src_val_tokenized_path, targ_val_tokenized_path)

# create list of bucket sizes
train_bucket_sizes = [len(train_set[b]['src']) for b in range(len(buckets))]
train_total_size = float(sum(train_bucket_sizes))

# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
# the size if i-th training bucket, as used later.
train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                       for i in xrange(len(train_bucket_sizes))]

print train_bucket_sizes, train_total_size

train_buckets_scale

batch_size = 150
def get_next_batch():
    random_number = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                     if train_buckets_scale[i] > random_number])
    
    bucket_size_permuation = np.random.permutation(train_bucket_sizes[bucket_id])
    
    train_encoder_batch = np.array([train_set[bucket_id]['src'][x] 
                            for x in bucket_size_permuation[:batch_size]])
    
    train_decoder_batch = np.array([train_set[bucket_id]['targ'][x] 
                            for x in bucket_size_permuation[:batch_size]])
    
    encoder_inputs_ = train_encoder_batch
    decoder_targets_ = train_decoder_batch
    
    encoder_inputs_length_ = buckets[bucket_id][0]
    decoder_targets_length_ = [buckets[bucket_id][1]]*batch_size
    
    return {
        encoder_inputs: encoder_inputs_,
        decoder_targets: decoder_targets_,
        decoder_sequence_length: decoder_targets_length_
    }

src_embed = []
with tf.gfile.GFile('data/src-embedding-lookup.txt', mode="rb") as src_embed_file:
    for line in src_embed_file:
        line = line.split()
        line_list = [float(x) for x in line]
        src_embed.append(line_list)

targ_embed = []
with tf.gfile.GFile('data/targ-embedding-lookup.txt', mode="rb") as targ_embed_file:
    for line in targ_embed_file:
        line = line.split()
        line_list = [float(x) for x in line]
        targ_embed.append(line_list)

num_epochs = 100000
max_batches=int(train_total_size/ batch_size)
for batch in range(max_batches):
        fd = get_next_batch()
        fd[src_embeddings] = src_embed
        fd[targ_embeddings] = targ_embed
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break