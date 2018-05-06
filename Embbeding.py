import numpy as np
import collections
import random
import pickle
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import math
from six.moves import xrange
import os.path
import DataProcessor as dp
import DataManager as dm

data_index = 0

# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window, words_ids):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  if data_index + span > len(words_ids):
    data_index = 0
  buffer.extend(words_ids[data_index:data_index + span])
  data_index += span
  for i in range(batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if data_index == len(words_ids):
      #buffer[:] = words_ids[:span]
      for id in words_ids[:span]:
        buffer.append(id)
      data_index = span
    else:
      buffer.append(words_ids[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(words_ids) - span) % len(words_ids)
  return batch, labels

def create_general_vocabulary(tokens):
    tags_copy = list(tokens)
    counter = collections.Counter(tags_copy)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    tags_copy, _ = list(zip(*count_pairs))
    vocabulary_to_id = dict(zip(tags_copy, range(len(tags_copy))))
    tags_ids = [vocabulary_to_id[tag] for tag in tokens]
    reverse_dictionary = dict([ (v, k) for k, v in vocabulary_to_id.items()])
    vocabulary = dm.Vocabulary(vocabulary_to_id, reverse_dictionary)
    return vocabulary, tags_ids

def createVocabulary(words, args):
    words_copy = list(words)
    counter = collections.Counter(words_copy)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    print("numbr of counted words", len(count_pairs))
    words_copy, number_ocurrences = list(zip(*count_pairs))
    if args.type_data in ['trim', 'native_trim'] and args.lower_limit > 0:
        words_copy = [words_copy[i] for i in range(len(number_ocurrences)) if number_ocurrences[i] > args.lower_limit]
        for i in range(len(words)):
            if words[i] not in words_copy:
                words[i] = '<tag_unknown>'
    print("numbr of words after applying lower limt", len(words_copy))
    vocabulary_to_id = dict(zip(words_copy, range(len(words_copy))))
    vocabulary_to_id['<tag_target_prep>'] = len(words_copy)
    vocabulary_to_id['<tag_unknown>'] = len(words_copy) + 1
    words_ids = [vocabulary_to_id[word] for word in words]
    reverse_dictionary = dict([ (v, k) for k, v in vocabulary_to_id.items()])
    vocabulary = dm.Vocabulary(vocabulary_to_id, reverse_dictionary)
    return vocabulary, words_ids

def returnEmbedding(vocabulary, words_ids, embedding_size, fname, num_sampled, args):
    if args.location == 'local':
        if os.path.isfile(fname + ".pickle"):
            return pickle.load(open(args.work_dir + fname + ".pickle", "rb"), encoding='latin1')
    else:
        if file_io.file_exists(args.work_dir + fname + ".pickle"):
            return pickle.load(file_io.FileIO(args.work_dir + fname + ".pickle", mode="r"))
    final_embeddings = embedFeature(len(vocabulary.value_to_id), embedding_size, words_ids, vocabulary.reverse, num_sampled, args)
    dp.saveData(args.work_dir + fname, final_embeddings)
    return final_embeddings

def getWordEmbedding(vocabulary, words_ids, args):
    fname = "data/Word_Embedding_Size_" + str(args.word_emb_size) + "_Window_" + str(args.size_window) + "_Skip_" \
            + str(args.skip_num) + "_Sampled_" + str(args.num_sampled) \
            + "_Limit_" + str(args.lower_limit) + "_Spelling_" + str(args.spelling)
    return returnEmbedding(vocabulary, words_ids, args.word_emb_size, fname, args.num_sampled, args)

def getTagEmbedding(tags_vocabulary, tags_ids, args):
    fname = "data/Embedding_Tags_Size_" + str(args.tags_emb_size) + "_Data_" + args.type_data
    num_sampled = len(tags_vocabulary.value_to_id)
    return returnEmbedding(tags_vocabulary, tags_ids, args.tags_emb_size, fname, num_sampled, args)

def getParseEmbedding(parse_vocabulary, parse_ids, args):
    fname = "data/Embedding_Parse__Size_" + str(args.index_emb_size) + "_Data_" + args.type_data
    num_sampled = len(parse_vocabulary.value_to_id)
    return returnEmbedding(parse_vocabulary, parse_ids, args.index_emb_size, fname, num_sampled, args)

def embedFeature(vocabulary_size, embedding_size, words_ids, reverse_dictionary, num_sampled, args):
    batch_size = 128
    #For visualization
    valid_size = 32
    valid_window = 40
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    graph = tf.Graph()
    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        # Explanation of the meaning of NCE loss:
        #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled= num_sampled,
                           num_classes=vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    # Step 5: Begin training.
    num_steps = 1000000

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(
                batch_size, args.skip_num, args.size_window, words_ids)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0
            #To visualize
            '''if step % 10000 == 0:
                sim = similarity.eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)'''
        final_embeddings = normalized_embeddings.eval()
    return final_embeddings
