# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
import numpy as np

class CaptionGenerator(object):
    def __init__(self, word_to_idx, V, dim_feature=97, dim_embed=512, dim_hidden=1024, n_time_step=16,
                 prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True):
        """
        Args:
            word_to_idx: word-to-index mapping dictionary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            n_time_step: (optional) Time step size of LSTM.
            prev2out: (optional) previously generated word to hidden state. (see Eq (2) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (2) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.iteritems()}
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = V
        self.D = dim_feature
        self.M = dim_embed
        self.H = dim_hidden
        self.T = n_time_step
        self._start = word_to_idx['<START>']
        self._end = word_to_idx['<END>']
        self._null = word_to_idx['<NULL>']

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

        # Place holder for features and labels
        self.features = tf.placeholder(tf.float32, [None, self.D])
        self.labels = tf.placeholder(tf.int32, [None, self.T+1])

        # sentence concat <start>
        self.sample_labels = tf.placeholder(tf.int32, [None, self.T+1])



    # def _get_initial_lstm(self, features):
    #     with tf.variable_scope('initial_lstm'):
    #         features_mean = tf.reduce_mean(features, 1)
    #         w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
    #         b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
    #         h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

    #         w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
    #         b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
    #         c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
    #         return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)
            features_proj = tf.reshape(features_proj, [-1, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.matmul(h, w) + b)  # (N, D)
            out_att = tf.matmul(h_att, w_att)  # (N, 1)
            alpha = tf.nn.softmax(out_att)
            context = features * alpha  # (N, D)
            return context, alpha

    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')  # (N, 1)
            context = tf.multiply(beta, context, name='selected_context')
            return context, beta

    def _get_encoder_layer(self, input_data,reuse=None):
        # Encoder embedding
        with tf.variable_scope('encode_lstm', reuse=reuse):
            encoder_embed_input = tf.contrib.layers.embed_sequence(input_data, self.V, self.M)
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.H)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(lstm_cell, encoder_embed_input, dtype=tf.float32)
        return encoder_output, encoder_state

    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits

    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x,
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode == 'train'),
                                            updates_collections=None,
                                            scope=(name + '_batch_norm'))

    def build_model(self):
        features = self.features
        labels = self.labels
        batch_size = tf.shape(features)[0]

        labels_in = labels[:, :self.T]
        labels_out = labels[:, 1:]
        # mask = tf.to_float(tf.not_equal(labels_out, self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='seq_features')

        _, (c, h) = self._get_encoder_layer(tf.cast(features, dtype=tf.int32)) # (N,M)
        x = self._word_embedding(inputs=labels_in) # (N,T,M)
        features_proj = self._project_features(features=features)

        loss = 0.0
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                import pdb; pdb.set_trace()
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x[:, t, :], context]), state=[c, h])

            logits = self._decode_lstm(x[:, t, :], h, context, dropout=self.dropout, reuse=(t != 0))
            x1 = tf.nn.softmax(logits)
            one_hot_result = tf.one_hot(labels_out[:, t], self.V)
            weight = tf.constant([1.0,5.0])
            softmax_results = tf.reduce_mean(-tf.reduce_sum(one_hot_result * weight * tf.log(x1), 1))
            loss += softmax_results

        if self.alpha_c > 0:
            alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)  # (N, D)
            alpha_reg = self.alpha_c * tf.reduce_sum((16. / 196 - alphas_all) ** 2)
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=7):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='seq_features')

        _, (c, h) = self._get_encoder_layer(tf.cast(features, dtype=tf.int32))
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], 0))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t != 0))
            sampled_word = tf.argmax(logits, 1)
            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, D)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        sampled_labels = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, betas, sampled_labels

    def build_loss(self):
        features = self.features
        labels = self.sample_labels
        # mask = tf.to_float(tf.not_equal(labels, self._null))

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='seq_features')

        c, h = self._get_initial_lstm(features=features)
        x = self._word_embedding(inputs=labels)
        features_proj = self._project_features(features=features)

        loss = []
        alpha_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)

        for t in range(self.T):
            if t == 0:
                word = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                word= x[:, t -1, :]
            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [word, context]), state=[c, h])

            logits = self._decode_lstm(word, h, context, reuse=(t != 0))
            softmax = tf.nn.softmax(logits, dim=-1, name=None)

            # loss.append( tf.transpose(tf.mul(tf.transpose(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(labels[:, t], 2), [1, 0]),  mask[:, t]), [1, 0]))
            loss.append(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(labels[:, t], 2))
        loss_out = tf.transpose(tf.stack(loss), (1, 0, 2))  # (N, T, max_len)

        return loss_out


    def build_multinomial_sampler(self, max_len=7):
        features = self.features

        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='seq_features')

        _, (c, h) = self._get_encoder_layer(tf.cast(features, dtype=tf.int32))
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.H)
        loss = []
        for t in range(self.T):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=True)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=(t != 0))
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=(t != 0))
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=(t != 0)):
                _, (c, h) = lstm_cell(inputs=tf.concat(1, [x, context]), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=(t != 0))
            softmax = tf.nn.softmax(logits, dim=-1, name=None)
            sampled_word = tf.multinomial(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)), 1)

            sampled_word = tf.reshape(sampled_word, [-1])
            loss.append(tf.log(tf.clip_by_value(softmax, 1e-20, 1.0)) * tf.one_hot(tf.identity(sampled_word), 2))

            sampled_word_list.append(sampled_word)

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))  # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))  # (N, T)
        loss_out = tf.transpose(tf.stack(loss), (1, 0, 2))  # (N, T, max_len)
        sampled_labels = tf.transpose(tf.stack(sampled_word_list), (1, 0))  # (N, max_len)
        return alphas, betas, sampled_labels,loss_out

