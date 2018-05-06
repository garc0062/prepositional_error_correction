# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 10:28:01 2018

@author: pablo
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#import packages
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import Embbeding as e
import pickle
import itertools
import argparse
import copy
#Customized packages
import DataProcessor as dp
import DataManager as dm
import ErrorCorrectionEvaluation as ce
import NN_models as nn
import math

FORWARD = 'fw'
BACKWARD = 'bw'

#Type_nn: ['simple', 'bidirectional', 'two_ways']
#ARGUMENTS
parser = argparse.ArgumentParser(description="Predicting Prepositions")
#model
parser.add_argument('--model_type', dest='model_type', help='type of NN', default='simple')
#size
parser.add_argument('--num_hidden', dest='num_hidden', type=int, help='Number of hidden', default=400)
parser.add_argument('--num_layer', dest='num_layer', type=int, help='type of layers', default=1)
#components
parser.add_argument('--dropout', dest='dropout', type=bool, help='If Dropout', default=False)
parser.add_argument('--keep_prob', dest='keep_prob', type=float, help='Dropout rate', default=1.0)
parser.add_argument('--attention', dest='attention', type=bool, help='If Attention', default=False)
#embedding and features
parser.add_argument('--size_window', dest='size_window', type=int, help='Size window to each size', default=2)
parser.add_argument('--skip_num', dest='skip_num', type=int, help='Number of times a word is used', default=1)
parser.add_argument('--num_sampled', dest='num_sampled', type=int, help='Number of negative examples to sample', default=64)
parser.add_argument('--word_emb_size', dest='word_emb_size', type=int, help='Size of words embedding', default=100)
parser.add_argument('--tags', dest='tags', type=bool, help='If tags as feature', default=False)
parser.add_argument('--tags_emb_size', dest='tags_emb_size', type=int, help='Size of tags embedding', default=20)
parser.add_argument('--parent_index', dest='parse', type=bool, help='If parse as feature', default=False)
parser.add_argument('--index_emb_size', dest='index_emb_size', type=int, help='Size of indexes embedding', default=10)
parser.add_argument('--lower_limit', dest='lower_limit', type=int, help='Lower limit of instances per tokens', default=0)
parser.add_argument('--under', dest='under', type=bool, help='If undersampling', default=False)
parser.add_argument('--over', dest='over', type=bool, help='If overfitting', default=False)
parser.add_argument('--over_pos', dest='over_pos', type=int, help='number of oversampling', default=0)
parser.add_argument('--under_pos', dest='under_pos', type=int, help='number of undersampling', default=0)
parser.add_argument('--type_input', dest='type_input', help='number of undersampling', default='embedding')
parser.add_argument('--type_data', dest='type_data', help='type of Data', default='trim')
parser.add_argument('--batch', dest='batch_size', type=int, help='Size of batch', default=1024)
parser.add_argument('--spelling', dest='spelling', type=bool, help='If correct spell', default=False)
parser.add_argument('--seq_limit', dest='seq_limit', type=int, help='Limit of sequence at each window side', default=5)
parser.add_argument('--original', dest='original', help='If preposition is included', default='exclude')
parser.add_argument('--class_limit', dest='class_limit', type=int, help='If preposition is included', default=10)
parser.add_argument('--not_include_classes', dest='not_include_classes', nargs='*', help='Classes not to include', default=[])
parser.add_argument('--epochs', dest='epochs', type=int, help='Epochs', default=80)
parser.add_argument('--step', dest='step',  type=int, help='number of step', default=-1)
#System
parser.add_argument('--job-dir', dest='job', help='unnecessary argument to make ml-engine works', default='')
parser.add_argument('--work', dest='work_dir', help='type of layers', default='')
parser.add_argument('--location', dest='location', help='To run locally', default='local')
parser.add_argument('--mode', dest='mode', help='mode of model (train/test/prod)', default='train')

args = parser.parse_args()
print(args.not_include_classes)
output_msgs = file_io.FileIO(args.work_dir + "output/" + dp.createName("log_", args) + ".txt", mode = "w")

# set variables for training
max_sentence_length = 2 * args.seq_limit + 1

#Get All training data
essays = dp.nuclesData(args, 'dev')
mistakes = dp.readNuclesMistakes(args.work_dir, 'dev')
mistakes = dp.cleanMistakes(mistakes)
#if args.mode == 'prod':
prod_essays = dp.nuclesData(args, 'prod')
#print(prod_essays)
prod_mistakes = dp.readNuclesMistakes(args.work_dir, 'prod')
prod_mistakes = dp.cleanMistakes(prod_mistakes)

print(len(prod_mistakes))
print(prod_mistakes)

if 'native' in args.type_data:
    native_words, _, native_essays = dp.readNativeData(args);

#Load information
#The data is devided into two: training and testing
training_size = int(0.9 * len(essays))
testing_size = len(essays) - training_size

i = iter(essays.items())
training_essays = dict(itertools.islice(i, training_size))
testing_essays = dict(i)

output_msgs.write("Loading data ... " + " " + str(len(training_essays)) + " " + str(len(testing_essays)) + " " + str(len(essays)) + "\n\t")

#Get data to embed words
#essays_corrected_preps = dp.correctPrepInEssays(mistakes, essays, 'Prep')
_, words, tags, parse = dp.getDataFromEssays(essays)

print("Number of words", len(words))

prep_classes, prep_top, all_prep_classes, count_prep = dp.getTopClassesByType(words, tags)
words_for_vocabulary = copy.deepcopy(words)
if 'native' in args.type_data:
    words_for_vocabulary.extend(native_words)
words_vocabulary, words_ids = e.createVocabulary(words_for_vocabulary, args)
#Name of preps from ids
reverse_preps = dict([ (v, k) for k, v in all_prep_classes.items()])
print("size of words vocabulary", words_vocabulary.length())

#Get embbedded words
final_embeddings = e.getWordEmbedding(words_vocabulary, words_ids, args)
print("Length word embedding", len(final_embeddings))
#Get embedded tags
final_embeddings_tags = []
#if args.tags:
tags_vocabulary, tags_ids = e.create_general_vocabulary(tags)
final_embeddings_tags = e.getTagEmbedding(tags_vocabulary, tags_ids, args)
#Get embedded parse
final_embeddings_parse = []
#if args.parse:
parse_vocabulary, parse_ids = e.create_general_vocabulary(parse)
final_embeddings_parse = e.getParseEmbedding(parse_vocabulary, parse_ids, args)

#Generate data for training
training_examples = dp.generateSamplesFromEssays("training", training_essays, mistakes, all_prep_classes,
                                                 words_vocabulary, tags_vocabulary, parse_vocabulary, args)
training_examples_with_errors = 0
training_errors_per_class = [0] * 100
for example in training_examples:
    if example.correct != example.original:
        training_errors_per_class[np.argmax(example.correct)] += 1
        training_examples_with_errors += 1
output_msgs.write("TOTAL ERRORS Training " + str(training_examples_with_errors) + "\n\t")
output_msgs.write("Errors per class " + str(training_errors_per_class) + "\n\t")

print("Training length", len(training_examples))
#Generate examples for testing
testing_examples = dp.generateSamplesFromEssays("testing", testing_essays, mistakes, all_prep_classes,
                                                words_vocabulary, tags_vocabulary, parse_vocabulary, args)
# Get prepositional errors
testing_examples_with_errors = 0
testing_errors_per_class = [0] * 100
for example in testing_examples:
    if example.correct != example.original:
        testing_errors_per_class[np.argmax(example.correct)] += 1
        #output_msgs.write(example.getEvaluation() + "ERROR TESTING \n\t")
        testing_examples_with_errors += 1
output_msgs.write("TOTAL ERRORS TESTING " + str(testing_examples_with_errors) + "\n\t")
output_msgs.write("Errors per class " + str(testing_errors_per_class) + "\n\t")

total_prep_sumes = 0
for i in range(30):
    print(testing_errors_per_class[i] + training_errors_per_class[i])
    total_prep_sumes+=testing_errors_per_class[i] + training_errors_per_class[i]
print("TOTAL", total_prep_sumes)

print("Testing length", len(testing_examples))
#generate Examples from native data
if 'native' in args.type_data:
    native_examples = dp.generateSamplesFromNativeData("native", native_essays, mistakes, all_prep_classes,
                                                       words_vocabulary, tags_vocabulary, parse_vocabulary, args)
    print("Native length", len(native_examples))
#if args.mode == 'prod':
prod_examples  = dp.generateSamplesFromEssays("prod", prod_essays, prod_mistakes, all_prep_classes,
                                              words_vocabulary, tags_vocabulary, parse_vocabulary, args)
print("Prod length", len(prod_examples))

#Remove Classes outside of target preps
training_examples = dp.removeExamplesOutsideTrainingTarget(training_examples, args)
print("Training length after removing classes outside", len(training_examples))
#Create validation data to do cross validation
training_examples, validation_examples = dp.divideExamples(training_examples, args)
print("Training length after validation", len(training_examples))
print("Validation length", len(validation_examples))

if args.under:
    training_examples = dp.underSample(training_examples, args)
    print("Training length after undersampling", len(training_examples))
if args.over:
    training_examples = dp.overSample(training_examples, args)
    print("Training length after oversampling", len(training_examples))

#Remove long sentences
#training_examples, training_correct_preps, training_student_preps, training_tags, training_tags_bw, training_tags_bw = dp.removeLongSents(training_examples, training_correct_preps, training_student_preps, training_tags, training_tags_bw, training_tags_bw, 106)
#testing_examples, testing_correct_preps, testing_student_preps, testing_tags, testing_examples_bw, testing_tags_bw = dp.removeLongSents(testing_examples, testing_correct_preps, testing_student_preps, testing_tags, testing_examples_bw, testing_tags_bw, 106)

#Balance Examples to have same size, a tenth class is introduced to represent other prepositions
#training_examples, training_examples_bw, training_student_preps, training_correct_preps, training_tags, training_tags_bw, training_parse, training_parse_bw = dp.balanceExamplesTwoWays(training_examples, training_examples_bw, training_student_preps, training_correct_preps, training_tags, training_tags_bw, training_parse, training_parse_bw, count_prep, args.class_limit)

#Define embedding
def embedding_sentences(raw_data):
    data = np.zeros((len(raw_data), max_sentence_length, args.word_emb_size))
    for sents_index in range(len(raw_data)):   
        sent_length = len(raw_data[sents_index])
        #print(sent_length)
        for word_index in range(sent_length):
            data[sents_index, word_index, :] = final_embeddings[raw_data[sents_index][word_index][0]]
    return data

def embedding_tags_sentences(raw_data):
    data = np.zeros((len(raw_data), max_sentence_length, args.tags_emb_size))
    for sents_index in range(len(raw_data)):
        sent_length = len(raw_data[sents_index])        
        for word_index in range(sent_length):
            data[sents_index, word_index, :] = final_embeddings_tags[raw_data[sents_index][word_index][0]]
    return data

def embedding_parse_sentences(raw_data):
    data = np.zeros((len(raw_data), max_sentence_length, args.index_emb_size))
    for sents_index in range(len(raw_data)):
        sent_length = len(raw_data[sents_index])
        for word_index in range(sent_length):
            data[sents_index, word_index, :] = final_embeddings_parse[raw_data[sents_index][word_index][0]]
    return data
def save_list(name, object):
    saving_lists = file_io.FileIO(dp.createName(args.work_dir + name, args) + ".pickle", mode = "w")
    pickle.dump(object, saving_lists, protocol=2)
    saving_lists.close()

def get_features(examples, type = FORWARD):
    if args.type_input == 'embedding':
        return join_embedding_features(examples, type)
    else:
        return join_one_hot_features(examples, type)

def join_one_hot_features(examples, type = FORWARD):
    feature = dp.one_hot([e.words for e in examples], words_vocabulary.length(), max_sentence_length)
    if args.tags:
        feature.extend(dp.one_hot([e.tags for e in examples], tags_vocabulary.length(), max_sentence_length))
    if args.parse:
        feature.extend(dp.one_hot([e.tags for e in examples], parse_vocabulary.length(), max_sentence_length))
    return feature

def join_embedding_features(examples, type = FORWARD):
    if type == 'fw':
        word_embeding_vector = embedding_sentences([e.words for e in examples])
        if args.tags:
            tag_embedding_vector = embedding_tags_sentences([e.tags for e in examples])
        if args.parse:
            parse_embedding_vector = embedding_parse_sentences([e.parse for e in examples])
    else:
        word_embeding_vector = embedding_sentences([e.words_bw for e in examples])
        if args.tags:
            tag_embedding_vector = embedding_tags_sentences([e.tags_bw for e in examples])
        if args.parse:
            parse_embedding_vector = embedding_parse_sentences([e.parse_bw for e in examples])
    features_embedded_vector = []
    for b_size in range(len(word_embeding_vector)):
        word_seq = word_embeding_vector[b_size][:]
        if args.tags:
            tag_seq = tag_embedding_vector[b_size][:]
        if args.parse:
            parse_seq = parse_embedding_vector[b_size][:]
        feature_seq = []
        for s_size in range(len(word_seq)):
            temp_final_seq = word_seq[s_size]
            if args.tags:
                temp_final_seq = np.append(temp_final_seq, tag_seq[s_size])
            if args.parse:
                temp_final_seq = np.append(temp_final_seq, parse_seq[s_size])
            feature_seq.append(temp_final_seq)
        features_embedded_vector.append(feature_seq)
    return features_embedded_vector

#Join Training examples with Native examples

if 'native' in args.type_data:
    training_examples.extend(native_examples)
    print("Training length after extending native examples", len(training_examples))

size_input = dm.getSizeInput(args)
if args.type_input == 'one_hot':
    size_input = words_vocabulary.length()
batch_inputs_bw = [[size_input*[0]]]

# MODEL
tf.reset_default_graph()
with tf.name_scope('Input'):
    sentences = tf.placeholder(tf.float32, [None, None, size_input], name="sentences")  # Batch X Sequence X Input
    sentences_bw = tf.placeholder(tf.float32, [None, None, size_input], name="sentences_bw")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    pos_pre = tf.placeholder(tf.int32, name="post_prep")
with tf.name_scope('Labels'):
    labels = tf.placeholder(tf.float32, [None, args.class_limit], "labels")
# get Predictor
preposition = nn.getModel(sentences=sentences, pos_pre= pos_pre, sentences_bw=sentences_bw, args=args)
# Optimize
#loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=preposition, labels=labels)
with tf.name_scope('Lost_Function'):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preposition, labels=labels, name="Softmax_Cross_Entropy")
    loss = tf.reduce_mean(loss, name="Mean")
    optimizer = tf.train.AdamOptimizer().minimize(loss, name="Optimizer")
#Evaluate Validation
#with tf.name_scope('Precision_Validation'):
labels_per_class = tf.placeholder(tf.int32, name="Labels_per_class")
predictions_per_class = tf.placeholder(tf.int32, name="Predictions_per_class")
auc = tf.metrics.auc(labels_per_class, predictions_per_class, num_thresholds=200, name="auc")
running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="auc")
running_vars_initializer = tf.variables_initializer(var_list=running_vars)
#Summarize
summary = tf.summary.FileWriter(args.work_dir + "summary/" + dp.createName("sum_", args), graph=tf.get_default_graph())
#tf.summary.scalar("cost_validation", loss)
tf.summary.scalar("auc", auc)
summary_op = tf.summary.merge_all()

if args.mode in ['test', 'prod']:
    num_layers = int(args.num_layer)
    hidden_size = int(args.num_hidden)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    '''if args.location == 'local':
        #saver = tf.train.import_meta_graph(dp.createName(args.work_dir + "models/predictor.ckpt.meta", args), clear_devices=True)
    else:
        saver = tf.train.Saver()'''
    saver = tf.train.Saver()
    saver.restore(session, dp.createName(args.work_dir + "models/predictor", args) + "_Step_" + str(args.step) + ".ckpt")
else:
    #TRAINIGN MODEL
    session = tf.Session()
    saver = tf.train.Saver(max_to_keep=None)
    session.run(tf.global_variables_initializer())
    num_steps = int(args.epochs * len(training_examples)/args.batch_size)
    print("Number of steps", num_steps)
    #Loss lists
    training_loss = []
    validation_loss = []
    validation_accuracy_list = []
    training_accuracy_list = []

    #Get prepositional errors
    validation_examples_with_errors = 0
    validation_errors_per_class = [0] * 100
    for example in testing_examples:
        if example.correct != example.original:
            validation_errors_per_class[np.argmax(example.correct)] += 1
            validation_examples_with_errors += 1
    output_msgs.write("TOTAL ERRORS TESTING " + str(validation_examples_with_errors) + "\n\t")
    output_msgs.write("Errors per class " + str(validation_errors_per_class) + "\n\t")

    validation_labels = [c.correct for c in validation_examples]
    validation_labels_indices = [np.argmax(p) for p in validation_labels]

    train_labels = [c.correct for c in training_examples]
    train_labels_indices = [np.argmax(p) for p in train_labels]

    save_list("lists/trainSize", len(training_examples))

    for step in range(num_steps):
        # get data for a batch
        offset = (step * args.batch_size) % (len(training_examples) - args.batch_size)
        batch_inputs = get_features(training_examples[offset: (offset + args.batch_size)], FORWARD)
        if args.type_data == 'two_ways':
            batch_inputs_bw = get_features(training_examples[offset: (offset + args.batch_size)], BACKWARD)
        batch_labels = [c.correct for c in training_examples[offset : (offset + args.batch_size)]]

        # put this data into a dictionary that we feed in when we run
        # the graph.  this data fills in the placeholders we made in the graph.
        data = {sentences: batch_inputs, sentences_bw: batch_inputs_bw, labels: batch_labels, keep_prob: args.keep_prob}
        #run the 'optimizer', 'loss', and 'pred_err' operations in the graph
        _, loss_value_train = session.run([optimizer, loss], feed_dict=data)

        # print stuff every 50 steps to see how we are doing
        if (step % 10 == 0):
            save_path = saver.save(session, dp.createName(args.work_dir + "models/predictor", args) + "_Step_" + str(step) + ".ckpt")
            training_loss.append(loss_value_train)
            output_msgs.write("Minibatch train loss at step " +  str(step) + ": " + str(loss_value_train) + "\n\t")
            #Validation loss function
            offset = 2000
            index = 0
            temp_validation_loss = []
            temp_validation_prep = []
            len_validation_examples = len(validation_examples)
            while index < len_validation_examples:
                if args.type_data == 'two_ways':
                    batch_inputs_bw = get_features(validation_examples[index:index + offset], BACKWARD)
                batch_inputs = get_features(validation_examples[index:index + offset], FORWARD)
                val_labels = [c.correct for c in validation_examples[index:index + offset]]
                data_validation = {sentences_bw: batch_inputs_bw, sentences:  batch_inputs, labels: val_labels, keep_prob: 1.0}
                val_prepositions, loss_value_validation = session.run([preposition, loss], feed_dict=data_validation)
                temp_validation_loss.append(loss_value_validation)
                temp_validation_prep.extend(val_prepositions)
                index = index + offset
            for i in range(len(validation_examples)):
                validation_examples[i].predicted = temp_validation_prep[i]
            session.run(running_vars_initializer)
            output_msgs.write("Validation " + str(step) + "\t\n")
            #ce.analyseClasses(validation_examples, labels_per_class, predictions_per_class, auc, session, output_msgs,
            #                  "valid" + str(step), args)
            ce.wasEvaluationSchema(validation_examples, output_msgs, "val_" + str(step), args)
            #Accuracy validation
            indices_validation = [np.argmax(p) for p in temp_validation_prep]
            sum_val = 0
            for i, l in zip(indices_validation, validation_labels_indices):
                if i == l:
                    sum_val += 1
            print("sum_val", sum_val, "Total", len(indices_validation))
            accuracy = float(sum_val)/float(len(indices_validation))
            validation_accuracy_list.append(accuracy)
            # Accuracy training
            index = 0
            temp_training_prep = []
            while index < len(training_examples):
                if args.type_data == 'two_ways':
                    batch_inputs_bw = get_features(training_examples[index:index + offset], BACKWARD)
                batch_inputs = get_features(training_examples[index:index + offset], FORWARD)
                val_labels = [c.correct for c in training_examples[index:index + offset]]
                data_validation = {sentences_bw: batch_inputs_bw, sentences:  batch_inputs, labels: val_labels, keep_prob: 1.0}
                train_prepositions = session.run(preposition, feed_dict=data_validation)
                temp_training_prep.extend(train_prepositions)
                index = index + offset
            indices_training = [np.argmax(p) for p in temp_training_prep]
            print("length training batch", len(indices_training))
            sum_val = 0
            for i, l in zip(indices_training, train_labels_indices):
                if i == l:
                    sum_val += 1
            print("sum_trainig", sum_val, "Total", len(indices_training))
            accuracy = float(sum_val) / float(len(indices_training))
            training_accuracy_list.append(accuracy)

            loss_value_validation = np.mean(temp_validation_loss)
            output_msgs.write("Validation: " + str(loss_value_validation) + "\n\t")
            output_msgs.flush()
            validation_loss.append(loss_value_validation)
            # Saving cross validation
            saving_lists = file_io.FileIO(dp.createName(args.work_dir + "lists/listTrainLoss", args) + ".pickle", mode = "w")
            pickle.dump(training_loss, saving_lists, protocol=2)
            saving_lists.close()
            saving_lists = file_io.FileIO(dp.createName(args.work_dir + "lists/listValLoss", args) + ".pickle", mode = "w")
            pickle.dump(validation_loss, saving_lists, protocol=2)
            saving_lists.close()
            saving_lists = file_io.FileIO(dp.createName(args.work_dir + "lists/listValAcc", args) + ".pickle", mode = "w")
            pickle.dump(validation_accuracy_list, saving_lists, protocol=2)
            saving_lists.close()
            saving_lists = file_io.FileIO(dp.createName(args.work_dir + "lists/listTrainAcc", args) + ".pickle", mode = "w")
            pickle.dump(training_accuracy_list, saving_lists, protocol=2)
            saving_lists.close()
            diff_train_val = math.fabs(loss_value_validation - loss_value_train)
            if diff_train_val > 1.0:
                print("Difference Vlidation Train", diff_train_val)
                break

if True:
    if args.mode == 'train':
        print(validation_loss)
        min_step = np.argmin(validation_loss) * 10
        output_msgs.write("Testing step (min)" + str(min_step))
        print("Testing step (min)", min_step)
        '''if args.location == 'local':
            saver = tf.train.import_meta_graph(dp.createName(args.work_dir + "models/predictor.ckpt.meta", args), clear_devices=True)
        else:'''
        saver = tf.train.Saver()
        saver.restore(session, dp.createName(args.work_dir + "models/predictor", args) + "_Step_" + str(min_step) + ".ckpt")


    #EVALUATING WITH TRAINING DATA
    offset = 2000
    index = 0
    '''
    training_prediction = []
    len_training_examples = len(training_examples)
    while index < len_training_examples:
        data_train = {sentences: join_features(embedding_sentences(training_examples[index:index + offset]),
                                               embedding_tags_sentences(training_tags[index:index + offset])),
                      labels: training_correct_preps[index:index + offset], keep_prob: 1.0}
        training_prediction_temp, loss_value_train = session.run([preposition, loss], feed_dict=data_train)
        training_prediction.extend(training_prediction_temp)
        output_msgs.write("Lost value TRAINING: " + str(loss_value_train) + "\n\t")
        index = index + offset
    #TRAINING DATA...
    output_msgs.write("TRAINING DATA..." + type_nn + " Hidden " + str(hidden_size) + " Layers " + str(num_layers) + "\n\t")
    ce.wasEvaluationSchema(training_student_preps, training_correct_preps, train_predictor_preps, output_msgs)
    ce.evaluateErrorDetection(training_student_preps, training_correct_preps, train_predictor_preps, output_msgs)
    ce.errorCorrectionBasedOnEdits(training_student_preps, training_correct_preps, train_predictor_preps, output_msgs)
    ce.wasEvaluationSchemaWithThreshold(training_student_preps, training_correct_preps, training_prediction, output_msgs)
    '''
    #TESTING WITH NEW DATA
    len_testing_examples = len(testing_examples)
    index = 0
    testing_prediction = []
    while index < len_testing_examples:
        if args.type_data == 'two_ways':
            batch_inputs_bw = get_features(testing_examples[index:index + offset], BACKWARD)
        batch_inputs = get_features(testing_examples[index:index + offset], FORWARD)
        data_test = {sentences_bw: batch_inputs_bw, sentences: batch_inputs, keep_prob: 1.0}
        testing_prediction_temp = session.run(preposition, feed_dict=data_test)
        testing_prediction.extend(testing_prediction_temp)
        index = index + offset
    for i in range(len(testing_prediction)):
        testing_examples[i].predicted = testing_prediction[i]


    output_msgs.write("EVALUATING ..." + dp.createName("", args) + "\n\t")

    output_msgs.write("Metrics Per Class for TESTING Data \n\t")
    session.run(running_vars_initializer)
    ce.analyseClasses(testing_examples, labels_per_class, predictions_per_class, auc, session, output_msgs, "testing", args)
    #summary.add_summary(summary_op)

    #TESTING DATA ...
    output_msgs.write("TESTING DATA ..." + dp.createName("", args) + "\n\t")
    print("Start testing")
    ce.wasEvaluationSchema(testing_examples, output_msgs, "test", args)
    print("finish was")
    ce.evaluateErrorDetection(testing_examples, output_msgs, "test", args)
    print("finish detection")
    ce.errorCorrectionBasedOnEdits(testing_examples, output_msgs, args)
    print("finish edits")

    output_msgs.write("Threshold: \n")
    ce.wasEvaluationSchemaWithThreshold(testing_examples, output_msgs, "test_a_th", args)
    ce.appliedPostAnalysis(testing_examples, output_msgs, "test", args)
    ce.appliedPostAnalysisDetection(testing_examples, output_msgs, "test_det", args)

    #TESTING RIGHT CLASSES ...
    output_msgs.write("TESTING RIGHT CLASSES ..." + dp.createName("", args) + "\n\t")
    testing_examples_limit = []

    for example in testing_examples:
        class_number = np.argmax(example.correct)
        if class_number < args.class_limit:
            testing_examples_limit.append(example)

    ce.wasEvaluationSchema(testing_examples_limit, output_msgs, "t_right", args)
    ce.evaluateErrorDetection(testing_examples_limit, output_msgs, "t_right", args)
    ce.errorCorrectionBasedOnEdits(testing_examples_limit, output_msgs, args)

    ce.appliedPostAnalysis(testing_examples_limit, output_msgs, "test", args)
    ce.appliedPostAnalysisDetection(testing_examples_limit, output_msgs, "test_det", args)

    output_msgs.write("Threshold: \t\n")
    ce.wasEvaluationSchemaWithThreshold(testing_examples_limit, output_msgs, "test_r_th", args)

    # TESTING WITH PRODUCTION DATA
    #if args.mode == 'prod':
    if True:
        len_prod_examples = len(prod_examples)
        index = 0
        prod_prediction = []
        while index < len_prod_examples:
            if args.type_data == 'two_ways':
                batch_inputs_bw = get_features(prod_examples[index:index + offset], BACKWARD)
            batch_inputs = get_features(prod_examples[index:index + offset], FORWARD)
            data_test = {sentences_bw: batch_inputs_bw, sentences: batch_inputs, keep_prob: 1.0}
            prod_prediction_temp = session.run(preposition, feed_dict=data_test)
            prod_prediction.extend(prod_prediction_temp)
            index = index + offset

        for i in range(len(prod_prediction)):
            prod_examples[i].predicted = prod_prediction[i]

        # Get prepositional error
        prod_examples_with_errors = 0
        prod_errors_per_class = [0] * 100
        for example in prod_examples:
            if example.correct != example.original:
                output_msgs.write(example.getEvaluation() + "ERROR PRODUCTION \n\t")
                prod_errors_per_class[np.argmax(example.correct)] += 1
                prod_examples_with_errors += 1
        output_msgs.write("TOTAL ERRORS PRODUCTION " + str(prod_examples_with_errors) + "\n\t")
        output_msgs.write("Errors per class " + str(prod_errors_per_class) + "\n\t")

        output_msgs.write("EVALUATING ..." + dp.createName("", args) + "\n\t")
        output_msgs.write("Metrics Per Class for Production Data \n\t" + str(testing_examples_with_errors))
        session.run(running_vars_initializer)
        ce.analyseClasses(prod_examples, labels_per_class, predictions_per_class, auc, session, output_msgs,
                          "prod", args)

        output_msgs.write(
            "EVALUATING PRODUCTION ..." + dp.createName("", args) + "\n\t")
        # train_predictor_preps = dp.get_one_hot_from_prob(training_prediction)
        # test_predictor_preps = dp.get_one_hot_from_prob(testing_prediction, args)

        # TESTING DATA ...
        output_msgs.write(
            "TESTING PRODUCTION ..." + dp.createName("", args) + "\n\t")
        ce.wasEvaluationSchema(prod_examples, output_msgs, "prod", args)
        ce.evaluateErrorDetection(prod_examples, output_msgs, "prod", args)
        ce.errorCorrectionBasedOnEdits(prod_examples, output_msgs, args)

        ce.appliedPostAnalysis(prod_examples, output_msgs, "prod", args)
        ce.appliedPostAnalysisDetection(prod_examples, output_msgs, "prod_det", args)

        output_msgs.write("Threshold: \n\t")
        ce.wasEvaluationSchemaWithThreshold(prod_examples, output_msgs, "prod_a_th", args)

        # TESTING RIGHT CLASSES ...
        output_msgs.write("TESTING PROD RIGHT CLASSES ..." + dp.createName("", args) + "\n\t")
        prod_examples_limit = []

        for example in prod_examples:
            class_number = np.argmax(example.correct)
            if class_number < args.class_limit:
                prod_examples_limit.append(example)

        ce.wasEvaluationSchema(prod_examples_limit, output_msgs, "p_right", args)
        ce.evaluateErrorDetection(prod_examples_limit, output_msgs, "p_right", args)
        ce.errorCorrectionBasedOnEdits(prod_examples_limit, output_msgs, args)

        output_msgs.write("Threshold Right: \n\t")
        ce.wasEvaluationSchemaWithThreshold(prod_examples_limit, output_msgs, "prod_r_th", args)


        threshold = 0.9
        def softmax(set_values):
            exp_values = np.exp(set_values - np.max(set_values))
            return exp_values / exp_values.sum()

        def compareDiffValues(examples, reverse_dictionary, all_prep_classes):
            size_samples = len(examples)
            for i in range(size_samples):
                student_preposition = np.argmax(examples[i].original)
                correct_preposition = np.argmax(examples[i].correct)
                predictor_preposition = np.argmax(examples[i].predicted)
                if student_preposition == correct_preposition and student_preposition == predictor_preposition:
                    continue

                predictions = softmax(examples[i].predicted)
                max_prediction = np.max(predictions)
                output_msgs.write(str(max_prediction) + "\n\t")
                #if(max_prediction < threshold):
                    #predictor_preposition = student_preposition

                output_msgs.write("Student: " + all_prep_classes[student_preposition] + " Correct: " + all_prep_classes[correct_preposition] + "\n\t")
                for j in range(args.class_limit):
                    output_msgs.write("Prediction for " + reverse_preps[j] + ': ' + str(predictions[j]) + "\n\t")

        '''compareDiffValues(testing_examples, words_vocabulary.reverse, reverse_preps)
        output_msgs.write("WITH LIMIT\n")
        compareDiffValues(testing_examples_limit, words_vocabulary.reverse, reverse_preps)

        output_msgs.write("PRODUCTION \n")
        compareDiffValues(prod_examples, words_vocabulary.reverse, reverse_preps)'''

output_msgs.close()