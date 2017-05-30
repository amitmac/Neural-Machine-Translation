
import gensim
import numpy as np
import tensorflow as tf
import re
import nltk

# regular expressions used to tokenize.
_PUNC_REPLACE_RE = re.compile(b"([.!?/\":_;)(&])")
_DIGIT_RE = re.compile(br"\d+")
_COMMA_RE = re.compile(",")

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# processed data path
src_train_processed_path = "data/processed/src-train.txt"
targ_train_processed_path = "data/processed/targ-train.txt"
src_val_processed_path = "data/processed/src-val.txt"
targ_val_processed_path = "data/processed/targ-val.txt"
src_test_processed_path = "data/processed/src-test.txt"
targ_test_processed_path = "data/processed/targ-test.txt"

"""
Word vectors paths
These word vectors are generated using fasttext using command
- ./fasttext print-word-vectors wiki.en.bin < src-train.txt > src-train-wordvec.txt

src-train-wordvec.txt file contains the word and its vector in each line
"""
src_train_wordvec = "data/wordvec/src-train-wordvec.txt"
targ_train_wordvec = "data/wordvec/targ-train-wordvec.txt"
src_val_wordvec = "data/wordvec/src-val-wordvec.txt"
targ_val_wordvec = "data/wordvec/targ-val-wordvec.txt"
src_test_wordvec = "data/wordvec/src-test-wordvec.txt"
targ_test_wordvec = "data/wordvec/targ-test-wordvec.txt"

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size=None):
    """
    Create vocabulary file (if it does not exist yet) from processed data files 
    generated using prepare_data module.
    
    """
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 1000 == 0:
                    print("  processing line %d" % counter)
                tokens = line.split()
                
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                        
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if max_vocabulary_size and (len(vocab_list) > max_vocabulary_size):
                vocab_list = vocab_list[:max_vocabulary_size]
            
            with tf.gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")

def initialize_vocabulary(vocab_path):
    vocab = {}
    rev_vocab = []
    with tf.gfile.GFile(vocab_path, mode="rb") as f:
        rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return (vocab, rev_vocab)

def get_wordvecs(wordvec_paths):
    wordvec = {}
    count = 0
    for wordvec_path in wordvec_paths:
        with tf.gfile.GFile(wordvec_path, mode="rb") as wordvec_file:
            for line in wordvec_file:
                line = line.split()
                if line[0] not in wordvec:
                    wordvec[line[0]] = line[1:]
                count += 1
    return wordvec

def create_embedding_lookup(embedding_path, vocab_path, wordvec_paths):
    vocab, rev_vocab = initialize_vocabulary(vocab_path)
    wordvec = get_wordvecs(wordvec_paths)
            
    with tf.gfile.GFile(embedding_path, mode="w") as embed_file:
        for word in rev_vocab:
            if word not in wordvec:
                wordvec[word] = np.random.uniform(-1,1,len(wordvec[wordvec.keys()[0]]))
            embed_file.write(" ".join([str(val) for val in wordvec[word]]) + "\n")
    print("Created word embedding lookup in %s" % embedding_path)

"""
Combine all the source data (train, val, test) and create a vocabulary
"""
src_vocab_path = 'data/src-vocab.txt'
src_all_data_path = 'data/processed/src-complete-data.txt'
with tf.gfile.GFile(src_train_processed_path, mode="rb") as src_train_file:
    with tf.gfile.GFile(src_val_processed_path, mode="rb") as src_val_file:
        with tf.gfile.GFile(src_test_processed_path, mode="rb") as src_test_file:
            src_all_data_list = src_train_file.readlines() + src_val_file.readlines() + src_test_file.readlines()
            with tf.gfile.GFile(src_all_data_path, mode="wb") as src_all_data_file:
                for line in src_all_data_list:
                    src_all_data_file.write(line + "\n")
                    
create_vocabulary(src_vocab_path, src_all_data_path)

"""
Combine all the target data (train, val, test) and create a vocabulary
"""
targ_vocab_path = 'data/targ-vocab.txt'
targ_all_data_path = 'data/processed/targ-complete-data.txt'
with tf.gfile.GFile(targ_train_processed_path, mode="rb") as targ_train_file:
    with tf.gfile.GFile(targ_val_processed_path, mode="rb") as targ_val_file:
        with tf.gfile.GFile(targ_test_processed_path, mode="rb") as targ_test_file:
            targ_all_data_list = targ_train_file.readlines() + targ_val_file.readlines() + targ_test_file.readlines()
            with tf.gfile.GFile(targ_all_data_path, mode="wb") as targ_all_data_file:
                for line in targ_all_data_list:
                    targ_all_data_file.write(line + "\n")
                    
create_vocabulary(targ_vocab_path, targ_all_data_path)

"""
Create source word embedding lookup
"""
src_embedding_path = "data/src-embedding-lookup.txt"
src_vocab_path = 'data/src-vocab.txt'
src_wordvec_paths = [src_train_wordvec, src_val_wordvec, src_test_wordvec]
create_embedding_lookup(src_embedding_path, src_vocab_path, src_wordvec_paths)

"""
Create target word embedding lookup
"""
targ_embedding_path = "data/targ-embedding-lookup.txt"
targ_vocab_path = 'data/targ-vocab.txt'
targ_wordvec_paths = [targ_train_wordvec, targ_val_wordvec, targ_test_wordvec]
create_embedding_lookup(targ_embedding_path, targ_vocab_path, targ_wordvec_paths)

def data_to_token_ids(data_path, target_path, vocab_path):
    vocab, rev_vocab = initialize_vocabulary(vocab_path)
    with tf.gfile.GFile(data_path, mode="rb") as f:
        with tf.gfile.GFile(target_path, mode="w") as tokens_file:
            for line in f:
                line = line.split()
                line_token_ids = [vocab.get(w, UNK_ID) for w in line]
                tokens_file.write(" ".join([str(tok) for tok in line_token_ids]) + "\n")

# Create tokens for training
src_train_tokenized_path = "data/tokenized/src-train-tokenized.txt"
targ_train_tokenized_path = "data/tokenized/targ-train-tokenized.txt"
data_to_token_ids(src_train_processed_path, src_train_tokenized_path, src_vocab_path)
data_to_token_ids(targ_train_processed_path, targ_train_tokenized_path, targ_vocab_path)

# Create tokens for validation
src_val_tokenized_path = "data/tokenized/src-val-tokenized.txt"
targ_val_tokenized_path = "data/tokenized/targ-val-tokenized.txt"
data_to_token_ids(src_val_processed_path, src_val_tokenized_path, src_vocab_path)
data_to_token_ids(targ_val_processed_path, targ_val_tokenized_path, targ_vocab_path)

# Create tokens for testing
src_test_tokenized_path = "data/tokenized/src-test-tokenized.txt"
targ_test_tokenized_path = "data/tokenized/targ-test-tokenized.txt"
data_to_token_ids(src_test_processed_path, src_test_tokenized_path, src_vocab_path)
data_to_token_ids(targ_test_processed_path, targ_test_tokenized_path, targ_vocab_path)

# Check number of unique tokens in source file
with tf.gfile.GFile(src_vocab_path, mode="rb") as src_vocab_file:
    print("Number of unique tokens in source file",len(src_vocab_file.readlines()))

# Check number of unique tokens in target file
with tf.gfile.GFile(targ_vocab_path, mode="rb") as targ_vocab_file:
    print("Number of unique tokens in source file",len(targ_vocab_file.readlines()))