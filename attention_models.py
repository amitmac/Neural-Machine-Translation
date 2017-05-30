import tensorflow as tf

class Bahadanau():
    def __init__(self, num_units):
        # Define the weights and biases for calculating the score of decoder hidden state and 
        # each encoder hidden state
        self.num_units = num_units
        self.W_score = tf.Variable(tf.random_uniform([num_units, num_units],-1,1),dtype=tf.float32)

        # Define weights for calculating the attentional hidden state
        self.W_c = tf.Variable(tf.random_uniform([2*num_units, num_units],-1,1),dtype=tf.float32)

    def bahadanau_model_single_step(self, previous_output, encoder_outputs):
        # calculate the score of the previous output from decoder with the encoder outputs for each time step
        # score = h_{t-1} * W_score * h_s
        # W_score - (decoder_hidden_units x decoder_hidden_units) - here 300x300
        # h_{t-1} - (batch_size x decoder_hidden_units)
        # h_s     - (batch_size x max_time_steps x encoder_hidden_units)
        decoder_hidden_units = self.num_units
        # h_{t-1} * W_score - (batch_size x decoder_hidden_units)
        # reshaped to (batch_size x decoder_hidden_units x 1)
        inter_score = tf.reshape(tf.matmul(previous_output, self.W_score),[-1,decoder_hidden_units,1])

        # h_s * inter_score - (batch_size x max_time_steps x 1)
        # here (decoder_hidden_units = encoder_hidden_units) 
        score = tf.matmul(encoder_outputs, inter_score)

        # reshape score to (batch_size x max_time_steps)
        batch_size, batch_max_time_steps, _ = tf.unstack(tf.shape(score))
        score = tf.reshape(score,[-1,batch_max_time_steps])

        # calculate alignment vector for each encoder_hidden_state
        # which equals to exp(score_i)/sum_{k=1}^{max_time_steps}(exp(score_{k})) for each single input
        # for a batch it would be (batch_size x max_time_steps)
        alignment_vector = tf.div(tf.exp(score),
                                    tf.reshape(tf.reduce_sum(tf.exp(score),1),[batch_size,1]))

        # calculate the weighted average of all the encoder hidden states according to alignment vector
        # for that first reshape alignment_vector to batch_size x max_time_steps x 1
        alignment_vector = tf.reshape(alignment_vector,[-1,batch_max_time_steps,1])
        context_vector = tf.reduce_mean(tf.multiply(encoder_outputs, alignment_vector),0)

        # reshape the context vector to (batch_size x decoder_hidden_units)
        context_vector = tf.reshape(context_vector,[-1,decoder_hidden_units])

        # calculate attentional hidden state by concatenating the context vector 
        # and decoder_previous_outout and multiplying by a weight
        concat_cv_po = tf.concat([context_vector, previous_output],axis=1)

        # calculate the attentional hidden state - (batch_size x decoder_hidden_units
        h_t_bar = tf.matmul(concat_cv_po, self.W_c) 

        return h_t_bar

    def bahadanau_model_multi_step(self, decoder_outputs, encoder_outputs):
        # read bahadanau_model_single_step for logic
        # the only difference is that output shape here is (batch_size, max_time_steps, num_units)
        decoder_hidden_units = self.num_units
        batch_size = tf.shape(encoder_outputs)[0]
        W_score_reshaped = tf.reshape(tf.tile(self.W_score, [batch_size ,1]),
                                      [-1,self.num_units, self.num_units])
        # inter_Score - (batch_size, max_time_steps, num_units
        inter_score = tf.multiply(decoder_outputs, W_score_reshaped)

        
        score = tf.matmul(encoder_outputs, inter_score)
        batch_size, batch_max_time_steps, _ = tf.unstack(tf.shape(score))
        score = tf.reshape(score,[-1,batch_max_time_steps])
        alignment_vector = tf.div(tf.exp(score),
                                    tf.reshape(tf.reduce_sum(tf.exp(score),1),[batch_size,1]))
        alignment_vector = tf.reshape(alignment_vector,[-1,batch_max_time_steps,1])
        context_vector = tf.reduce_mean(tf.multiply(encoder_outputs, alignment_vector),0)
        context_vector = tf.reshape(context_vector,[-1,decoder_hidden_units])
        concat_cv_po = tf.concat([context_vector, previous_output],axis=1)
        h_t_bar = tf.matmul(concat_cv_po, self.W_c) 

        return h_t_bar