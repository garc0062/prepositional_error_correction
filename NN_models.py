# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 18:12:29 2018

@author: pablo
"""

import tensorflow as tf
import math

class MultipleLayerRNN:
    
    def __init__(self, num_layers, type_cell = 'LSTM'):
        self.type_cell = type_cell
        self.num_layers = num_layers
        self.cells = []
        
    def createCells(self, dropout = False, attention = False, attn_length = 0, keep_prob=0.8, hidden_size = 400):
        for i in range(self.num_layers):
            if self.type_cell == 'LSTM':
                temp_cell = tf.contrib.rnn.LSTMCell(hidden_size)
                if dropout and (i != 0 or i!= (self.num_layers - 1)):
                    temp_cell = tf.nn.rnn_cell.DropoutWrapper(temp_cell, output_keep_prob=keep_prob)
                if attention:
                    temp_cell = tf.contrib.rnn.AttentionCellWrapper(temp_cell, attn_length=attn_length, state_is_tuple=True)
                self.cells.append(temp_cell)
        #self.multicell = tf.contrib.rnn.MultiRNNCell(cells=self.cells , state_is_tuple=True)

'''
It defines a logistic component.
Use to define outputs from 
'''
def linear(input_, output_size, name, pos_target = -1, init_bias=0.0):
    print(init_bias)
    shape = input_.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable("weights", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
    if init_bias is None:
        print('None')
        return tf.matmul(input_, W)
    with tf.variable_scope(name):
        print('With bias')
        b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
    print(input_)
    print(W)
    return tf.matmul(input_, W) + b

'''
Function to get the length of the examples on the fly
'''
def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length

'''
Return predictor according to type. Model is set according to arguments.
'''
def getModel(sentences, pos_pre, sentences_bw, args):
    if args.model_type == 'simple':
        return getSimpleLSTMPredictor(sentences, args)
    elif args.model_type == 'bi':
        return getBidirectionalLSTMPredictor(sentences, pos_pre, args)
    elif args.model_type == 'two_ways':
        return getTwoWaysLSTMPredictor(sentences, sentences_bw, args)

'''
Return predictor with a simple LSTM as core.
'''
def getSimpleLSTMPredictor(sentences, args):
    # MODEL
    multiLayerLSTM = MultipleLayerRNN(num_layers=args.num_layer, type_cell='LSTM')
    multiLayerLSTM.createCells(dropout = args.dropout, attention = args.attention, attn_length = args.seq_limit,
                               keep_prob = args.keep_prob, hidden_size = args.num_hidden)
    multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=multiLayerLSTM.cells, state_is_tuple=True)

    _, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, sentences, sequence_length=length(sentences), dtype=tf.float32)
    # Output
    preposition = linear(final_state[-1][-1], args.class_limit, name="output", pos_target = -1)
    print(preposition)
    return preposition

'''
Model for Embedded_nucles_full_sentence_without_det_adjectives_lstm_balance_data_with_Tags
'''
def getBidirectionalLSTMPredictor(sentences, pos_pre, args):
    # MODEL
    multiLayerLSTM_FW = MultipleLayerRNN(num_layers=args.num_layer, type_cell='LSTM')
    multiLayerLSTM_FW.createCells(dropout = args.dropout, attention = args.attention, attn_length = args.seq_limit,
                                  keep_prob = args.keep_prob, hidden_size = args.num_hidden)

    multiLayerLSTM_BW = MultipleLayerRNN(num_layers=args.num_layer, type_cell='LSTM')
    multiLayerLSTM_BW.createCells(dropout = args.dropout, attention = args.attention, attn_length = args.seq_limit,
                                  keep_prob = args.keep_prob, hidden_size = args.num_hidden)

    multi_lstm_cells_fw = tf.contrib.rnn.MultiRNNCell(cells=multiLayerLSTM_FW.cells, state_is_tuple=True)
    multi_lstm_cells_bw = tf.contrib.rnn.MultiRNNCell(cells=multiLayerLSTM_BW.cells, state_is_tuple=True)

    _, final_state_tuple = tf.nn.bidirectional_dynamic_rnn(cell_fw=multi_lstm_cells_fw,
                                                           cell_bw=multi_lstm_cells_bw, inputs=sentences,
                                                           sequence_length=length(sentences),
                                                           dtype=tf.float32)
    out_fw, out_bw = final_state_tuple

    #final_state = tf.reduce_mean([out_fw, out_bw], 0)
    print("One state", out_fw, "The other", out_bw)
    final_state = tf.concat([out_fw, out_bw], axis=-1)

    # Output
    pos_pre = args.seq_limit
    preposition = linear(final_state[-1][-1], args.class_limit, name="output", pos_target=pos_pre)
    return preposition
'''
Two ways model
'''
def getTwoWaysLSTMPredictor(sentences_fw, sentences_bw, pos_pre, args):
    multiLayerLSTM_FW = MultipleLayerRNN(num_layers=args.num_layer, type_cell='LSTM')
    multiLayerLSTM_FW.createCells(dropout = args.dropout, attention = args.attention, attn_length = args.seq_limit,
                                  keep_prob = args.keep_prob, hidden_size = args.num_hidden)

    multiLayerLSTM_BW = MultipleLayerRNN(num_layers=args.num_layer, type_cell='LSTM')
    multiLayerLSTM_BW.createCells(dropout = args.dropout, attention = args.attention, attn_length = args.seq_limit,
                                  keep_prob = args.keep_prob, hidden_size = args.num_hidden)

    multi_lstm_cells_fw = tf.contrib.rnn.MultiRNNCell(cells=multiLayerLSTM_FW.cells, state_is_tuple=True)
    multi_lstm_cells_bw = tf.contrib.rnn.MultiRNNCell(cells=multiLayerLSTM_BW.cells, state_is_tuple=True)

    with tf.variable_scope('fw'):
        out_dy_rnn_fw, state_dy_rnn_fw = tf.nn.dynamic_rnn(multi_lstm_cells_fw, sentences_fw,
                                                           sequence_length=length(sentences_fw), dtype=tf.float32)
    with tf.variable_scope('bw'):
        out_dy_rnn_bw, state_dy_rnn_bw = tf.nn.dynamic_rnn(multi_lstm_cells_bw, sentences_bw,
                                                           sequence_length=length(sentences_bw), dtype=tf.float32)

    #final_state = tf.reduce_mean([state_dy_rnn_fw, state_dy_rnn_bw], 0)
    final_state = tf.concat([state_dy_rnn_fw[-1][-1], state_dy_rnn_bw[-1][-1]], axis=-1)

    # Output
    #preposition = linear(final_state[-1][-1], 10, name="output")
    preposition = linear(final_state[-1][-1], args.class_limit, name="output", pos_target=pos_pre)
    return preposition