import gensim
import numpy as np
import tensorflow as tf
import re
import nltk

# raw data path
src_train_path = "data/raw/src-train.txt"
targ_train_path = "data/raw/targ-train.txt"
src_val_path = "data/raw/src-val.txt"
targ_val_path = "data/raw/targ-val.txt"
src_test_path = "data/raw/src-test.txt"
targ_test_path = "data/raw/targ-test.txt"

# processed data path
src_train_processed_path = "data/processed/src-train.txt"
targ_train_processed_path = "data/processed/targ-train.txt"
src_val_processed_path = "data/processed/src-val.txt"
targ_val_processed_path = "data/processed/targ-val.txt"
src_test_processed_path = "data/processed/src-test.txt"
targ_test_processed_path = "data/processed/targ-test.txt"

# regular expressions used to tokenize.
_PUNC_REPLACE_RE = re.compile(b"([.!?/\":_;)(&])")
_DIGIT_RE = re.compile(br"\d+")
_COMMA_RE = re.compile(",")

def process_data(data_path, processed_data_path, tokenizer=None, normalize_digits=True):
    with tf.gfile.GFile(data_path, mode="rb") as src_file:
        with tf.gfile.GFile(processed_data_path, mode="w") as src_processed_file:
            count = 0
            for line in src_file:
                count += 1
                if count%2000 == 0:
                    print("%d sentences processed" % count)
                line = line.strip().lower()
                line = _PUNC_REPLACE_RE.sub("",line)
                line = _COMMA_RE.sub(" , ",line)
                if normalize_digits:
                    line = _DIGIT_RE.sub("0",line)
                src_processed_file.write(line + "\n")

print("Processing source train data...")
process_data(src_train_path, src_train_processed_path)
print("Processing target train data...")
process_data(targ_train_path, targ_train_processed_path)
print("Processing source val data...")
process_data(src_val_path, src_val_processed_path)
print("Processing target val data...")
process_data(targ_val_path, targ_val_processed_path)
print("Processing source test data...")
process_data(src_test_path, src_test_processed_path)
print("Processing target test data...")
process_data(targ_test_path, targ_test_processed_path)

# Now use the pretrained wordvec model to generate word vector of each word in the files generated
# For fastext use the below command
# $ ./fasttext print-word-vectors wiki.en.bin < src-train.txt > src-train-wordvec.txt