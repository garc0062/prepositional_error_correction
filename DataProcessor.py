import nltk
import numpy as np
import collections
import copy
from random import shuffle
from tensorflow.python.lib.io import file_io
from autocorrect import spell
import pickle
import os.path
import DataManager as dm
import random

NONE = -1
NO_VALID = -2
INSERTION = 1
MISSING = 2
WRONG_CHOICE = 3

nltk.download('averaged_perceptron_tagger')

'''
Read data from the NUCLE corpus and returns all words and all sentences as an array.
It also includes essays as dictionaties: in each element arrays of paraphraphs, in which
arrays of sentences are included. Each sentence is a dictionay containing an array of
words and rags respectively.
'''
def nuclesData(args, mode):
    fname = args.work_dir + "data/nucle_essays_spelling_" + str(args.spelling)
    if args.mode == 'prod':
        fname = args.work_dir + "data/prod_essays_spelling_" + str(args.spelling)
    print(fname)
    if args.location == 'local':
        if os.path.isfile(fname + ".pickle"):
            return pickle.load(open(fname + ".pickle", "rb"), encoding='latin1')
    else:
        if file_io.file_exists(fname + ".pickle"):
            return pickle.load(file_io.FileIO(fname + ".pickle", mode="r"))

    if mode == 'dev':
        f = file_io.FileIO(args.work_dir + "data/conll13st-preprocessed.conll", mode="r")
        prevEssayId = 829
    elif mode == 'prod':
        f = file_io.FileIO(args.work_dir + "data/official-preprocessed.conll", mode="r")
        prevEssayId = 2

    #Previous IDS
    prevParagId = 0
    prevSentId = 0
    #global IDS
    essay_ids = []
    parag_ids = []
    sents_ids = []
    words_ids = []
    #General Information
    words = []
    tags = []
    parse = []
    sents = []
    #structured information
    wordsTemp = []
    tagsTemp = []
    parseTemp = []
    sentsTemp = []
    paragsTemp = []
    essays = {}

    next = f.readline()
    while next != "":
        splitted = next.strip().split("\t")
        next = f.readline()
        if len(splitted) < 5:
            continue
        essay_ids.append(splitted[0])
        parag_ids.append(splitted[1])
        sents_ids.append(splitted[2])
        words_ids.append(splitted[3])
        word = splitted[4]
        if args.spelling and ('.' not in word and '/' not in word):
            if '-' in word:
                min_words = word.strip().split("-")
                word = ""
                for min_w in min_words:
                    word = word + spell(min_w)
            else:
                word = spell(word)
        words.append(word.lower())
        wordsTemp.append(word.lower())
        if splitted[6] == '-':
            parse.append(1)
            parseTemp.append(1)
        else:
            parse.append(int(splitted[6]) + 1)
            parseTemp.append(int(splitted[6]) + 1)
        #print(splitted[5])
        if splitted[5] == 'TO' and splitted[7] == 'prep':
            tags.append('IN')
            tagsTemp.append('IN')
        else:
            tags.append(splitted[5])
            tagsTemp.append(splitted[5])
        #validata each row
        if(prevSentId != int(sents_ids[-1]) or prevParagId != int(parag_ids[-1])):
            tt = tagsTemp.pop()
            tw = wordsTemp.pop()
            tp = parseTemp.pop()
            sentsTemp.append({'words': wordsTemp, 'tags': tagsTemp, 'parse': parseTemp})
            wordsTemp = [tw]
            tagsTemp = [tt]
            parseTemp = [tp]
            prevSentId = int(sents_ids[-1])
        if(prevParagId != int(parag_ids[-1]) or prevEssayId != int(essay_ids[-1])):
            paragsTemp.append(sentsTemp)
            sents.extend(sentsTemp)
            sentsTemp = []
            prevParagId = int(parag_ids[-1])
        if(prevEssayId != int(essay_ids[-1])):
            essays.update({essay_ids[-2]:paragsTemp})
            paragsTemp = []
            prevEssayId = int(essay_ids[-1])      
    sentsTemp.append({'words': wordsTemp, 'tags': tagsTemp, 'parse': parseTemp})
    paragsTemp.append(sentsTemp)
    essays.update({essay_ids[-1]:paragsTemp})

    saveData(fname, essays)
    return essays

'''
Read Native data
'''
def readNativeData(args):
    native_data = []
    if args.type_data in ['native', 'native_trim']:
        native_data = restoreData(args.work_dir + "data/native_corpus.pickle", args)
    words = []
    tags = []
    for sent in native_data:
        for word in sent['words']:
            words.append(word.lower())
        tags.extend(sent['tags'])
    return words, tags, native_data


'''
Save data
'''
def saveData(fname, object):
    pickle_in = file_io.FileIO(fname + ".pickle", mode="w")
    pickle.dump(object, pickle_in, protocol=2)
    pickle_in.close()

'''
Restore data
'''
def restoreData(fname, args):
    if args.location == 'local':
        object = pickle.load(open(fname, "rb"), encoding='latin1')
    else:
        pickle_in = file_io.FileIO(fname, mode="r")
        object = pickle.load(pickle_in)
    return object

'''
Reads mistakes from the NUCLE corpus.
Returns an array of dictionaries with:
    nid: essay ID
    pid: paragraph id
    sid: sentence ID
    start_token
    end_token
'''
def readNuclesMistakes(work_dir, mode):
    from lxml import etree
    if mode == 'dev':
        f = file_io.FileIO(work_dir + "data/conll13st-preprocessed.conll.ann", mode="r")
    elif mode == 'prod':
        f = file_io.FileIO(work_dir + "data/official-preprocessed.conll.ann", mode="r")
    mistakes = []
    
    doc = etree.parse(f)
    listAnn = doc.getroot()
    for ann in listAnn:
        misAnn = ann.getchildren()
        for mis in misAnn:
            #print(mis.get('nid'))
            mistakes.append({'nid': mis.get('nid'), 
                             'pid': int(mis.get('pid')), 
                             'sid': int(mis.get('sid')), 
                             'start_token': int(mis.get('start_token')),
                             'end_token': int(mis.get('end_token'))
                            })
            inf = mis.getchildren()
            for detail in inf:
                mistakes[-1].update({detail.tag: detail.text})                
                #print(detail.tag)
    return mistakes

def cleanMistakes(mistakes):
    prep_mistakes = []
    for m in mistakes:
        if m['TYPE'] == 'Prep':
            prep_mistakes.append(m)
    return prep_mistakes
    

#Removes elements from start to end
def remove(sent, start, end):
    index = start
    while index < end:
        del sent[start]
        index = index + 1
    return sent

#Insert elements in 'corrections' from 'start' into 'sent'
def insert(sent, start, correctiones):
    for correction in correctiones:
        sent.insert(start, correction)
        start = start + 1
    return sent

#Get tags from list 'sents' (words)
def getTags(sent):
    pos_tags = nltk.pos_tag(sent)
    tags = []
    for pos_tag in pos_tags:
        tags.append(pos_tag[1])
    for i in range(len(tags)):
        if tags[i] == 'TO':
            tags[i]= 'IN'
    return tags

#Correct errors in essays by type. Default 'Prep'
def correctPrepInEssays(mistakes, essays_to_keep, typeError = 'Prep'):
    essays= copy.deepcopy(essays_to_keep)
    for mistake in mistakes:
        if mistake['TYPE'] == typeError:
            wrong = essays[mistake['nid']][mistake['pid']][mistake['sid']]['words']
            tags = essays[mistake['nid']][mistake['pid']][mistake['sid']]['tags']
            right = wrong

            #we remove whatever is wrong
            right = remove(wrong, mistake['start_token'], mistake['end_token'])
            tags = remove(tags, mistake['start_token'], mistake['end_token'])

            #Only insert corrections if there is any
            if mistake['CORRECTION'] != None:
                corrections = mistake['CORRECTION'].split()
                right = insert(wrong, mistake['start_token'], corrections)
                tags = insert(tags, mistake['start_token'], getTags(corrections))           
    return essays

'''Given n essays, it returns their sentences, words and tags.'''
def getDataFromEssays(essays):
    tags = []
    sents = []
    words = []
    parse = []
    essay_keys = list(essays.keys())
    for essay_key in essay_keys:
        parags = essays[essay_key][:]
        for parag in parags:
            temp_sents = parag[:]
            for sent in temp_sents:
                sents.append(sent['words'])
                words.extend(sent['words'])
                tags.extend(sent['tags'])
                parse.extend(sent['parse'])
    return sents, words, tags, parse

def getTopClassesByType(words, tags, typeTag = 'IN', numTop = 10):
    preps = []
    preps = [prep.lower() for prep, tag in zip(words, tags) if tag==typeTag and prep.lower() not in preps]
    print(len(preps))

    counter = collections.Counter(preps)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    prep_classes = {}
    prep_top = []
    all_classes = {}
    for i in range(len(count_pairs)):
        all_classes[count_pairs[i][0]] = i
    for i in range(numTop):
        prep_classes[count_pairs[i][0]] = i
        prep_top.append(count_pairs[i][0])
    return prep_classes, prep_top, all_classes, count_pairs

def createParseVocabulary(parse, args):
    words_copy = list(parse)
    counter = collections.Counter(words_copy)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    print(count_pairs[0:100])
    words_copy, number_ocurrences = list(zip(*count_pairs))
    print(len(words_copy))
    vocabulary_to_id = dict(zip(words_copy, range(len(words_copy))))
    words_ids = [vocabulary_to_id[word] for word in parse]
    reverse_dictionary = dict([ (v, k) for k, v in vocabulary_to_id.items()])
    return vocabulary_to_id, reverse_dictionary, words_ids

def generateTrainingData(sents, words, tags, classes, top_classes, vocabulary_to_id):    
    ti  = []
    to = []
    begin = 0
    for i in range(0, len(sents)):
        prep_pos = -1
        end = len(sents[i]) + begin
        for j in range(begin, end):
            if(tags[j]=='IN' and words[j] in top_classes):
                prep_pos = j
                break
        if prep_pos == -1: 
            begin = end
            continue
        temp_list = []
        for j in range(1, 6):
            if(prep_pos - j >= begin):
                temp_list.insert(0, [int(vocabulary_to_id[words[prep_pos - j].lower()])])
            else:
                temp_list.insert(0, [int(vocabulary_to_id['<tag_empty>'])])
            if prep_pos + j < end and tags[prep_pos + j] != 'IN':
                    temp_list.insert(len(temp_list) + 1, [int(vocabulary_to_id[words[prep_pos + j].lower()])])
            else:
                temp_list.insert(len(temp_list) + 1, [int(vocabulary_to_id['<tag_empty>'])])
        ti.append(np.array(temp_list))
        begin = end 
        temp_list = ([0]*10)
        temp_list[classes[words[prep_pos]]] = 1
        to.append(temp_list)
    return ti, to

'''Generates testing data'''
#def generateTestingData(essays, mistakes)

def one_hot(raw_data, vocab_size, size_seq):
    print(len(raw_data))
    print(size_seq)
    data = np.zeros((len(raw_data), size_seq, vocab_size))
    for phrase_index in range(len(raw_data)):
        phrase = raw_data[phrase_index]
        print(len(phrase))
        for word_index in range(len(phrase)):
            word_id = phrase[word_index]
            data[phrase_index, word_index, word_id] = 1
    return data

def devideRandomlyTrainingAndTestingData(raw_input, raw_output):
    EXAMPLES_SIZE = len(raw_input)
    TRAINING_SIZE = EXAMPLES_SIZE - int(0.1*EXAMPLES_SIZE)

    rand_ids = [x for x in range(EXAMPLES_SIZE)]
    shuffle(rand_ids)

    train_input = [None] * TRAINING_SIZE
    train_output = [None] * TRAINING_SIZE

    test_input = [None] * (EXAMPLES_SIZE - TRAINING_SIZE)
    test_output = [None] * (EXAMPLES_SIZE - TRAINING_SIZE)

    for x in range(0, TRAINING_SIZE):
        train_input[x] = raw_input[rand_ids[x]]
        train_output[x] = raw_output[rand_ids[x]]   

    for x in range(0, (EXAMPLES_SIZE - TRAINING_SIZE)):
        test_input[x] = raw_input[rand_ids[x + TRAINING_SIZE]]
        test_output[x] = raw_output[rand_ids[x + TRAINING_SIZE]]   
    return train_input, train_output, test_input, test_output

def hasPrep(sent):
    temp_tags = getTags(sent)
    for temp_tag in temp_tags:
        if temp_tag == 'IN':
            return 1
    return 0

def identifyPrepPosition(sent):
    prep_pos = []
    words = sent['words']
    tags = sent['tags']
    for j in range(len(words)):
        if(tags[j]=='IN'):
            prep_pos.append(j)
    return prep_pos

'''
Returns corrected preposition if error exists and it is wrong choice. Otherwise returns empty array.
Additionally return the type of prepositional error
Applies only for insertion and wrong choice as input position assumes that preposition exists in original text.
'''
def getCorrectPrep(essay_key, parag_id, sent_id, mistakes, position, classes_to_id):
    type_error = NONE
    right_prep = []
    for mistake in mistakes:
        if mistake['TYPE'] == 'Prep':
            if mistake['nid'] == essay_key and mistake['pid'] == parag_id and mistake['sid'] == sent_id:
                if position >= mistake['start_token'] and position <= mistake['end_token']:
                    if mistake['CORRECTION'] != None:
                        corrections = mistake['CORRECTION'].split()
                        if(len(corrections) == 1 and (hasPrep(corrections))):
                            type_error = WRONG_CHOICE
                            if corrections[0] in classes_to_id:                            
                                right_prep = ([0]*len(classes_to_id))
                                right_prep[classes_to_id[corrections[0]]] = 1
                            else:
                                type_error = NO_VALID
                        else:
                            type_error = INSERTION
                    else:
                        type_error = INSERTION      
                break        
    return right_prep, type_error

'''
Generate examples from native data
'''
def generateSamplesFromNativeData(native_essays, args, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id):
    examples = []
    examples_bw = []
    tags_examples = []
    tags_examples_bw = []
    student_preps = []
    correct_preps = []
    parse_examples = []
    parse_examples_bw = []
    for sent in native_essays:
        positions = sent['positions']
        for position in positions:
            words = sent['words']
            if words[position] not in classes_to_id:
                continue
            temp_sample, temp_student, temp_tags, temp_p, temp_sample_bw, temp_tags_bw, temp_p_bw = eval(getGenerator(
                args.type_nn) + "(sent, position, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id, args.original, args.window)")
            temp_right = temp_student
            examples.append(temp_sample)
            student_preps.append(temp_student)
            correct_preps.append(temp_right)
            tags_examples.append(temp_tags)
            tags_examples_bw.append(temp_tags_bw)
            examples_bw.append(temp_sample_bw)
            parse_examples.append(temp_p)
            parse_examples_bw.append(temp_p_bw)
    return examples, student_preps, correct_preps, tags_examples, parse_examples, examples_bw, tags_examples_bw, parse_examples_bw

'''
Return data if exists
'''
def restoreObject(fname, args):
    if args.location == 'local':
        if os.path.isfile(fname + ".pickle"):
            return pickle.load(open(args.work_dir + fname + ".pickle", "rb"), encoding='latin1')
    else:
        if file_io.file_exists(args.work_dir + fname + ".pickle"):
            return pickle.load(file_io.FileIO(args.work_dir + fname + ".pickle", mode="r"))
    return 0

'''
Returns examples.
'''
def generateSamplesFromEssays(type, essays, mistakes, classes_to_id, words_vocabulary, tags_vocabulary, parse_vocabulary, args):
    fname = createNameForExamples("data/" + type + "_examples_", args)
    examples = restoreObject(fname, args)
    if examples != 0:
        return examples

    examples = []
    #Unroll essays down to sentences
    essay_keys = list(essays.keys())
    for essay_key in essay_keys:
        parags = essays[essay_key][:]
        for parag_id in range(len(parags)):
            parag = parags[parag_id]
            temp_sents = parag[:]
            for sent_id in range(len(temp_sents)):
                sent = temp_sents[sent_id]
                #Per each sentence get samples
                #First get position of preposition
                pre_posts =  identifyPrepPosition(sent)
                #Second, per each position, generate samples
                for position in pre_posts:
                    temp_right, type_error = getCorrectPrep(essay_key, parag_id, sent_id, mistakes, position, classes_to_id)
                    #Check to add samples
                    if(type_error == NONE or type_error == WRONG_CHOICE):
                        #temp_sample, temp_student, temp_tags, temp_parse, temp_sample_bw, temp_tags_bw, temp_parse_bw = eval(getGenerator(type_examples) + "(sent, position, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id, original, max_size)")
                        temp_example = generateSamplesFromSent(sent, position, classes_to_id, words_vocabulary, tags_vocabulary, parse_vocabulary, args)

                        if type_error == NONE:
                            temp_example.correct = temp_example.original
                        else:
                            temp_example.correct = temp_right
                        examples.append(temp_example)
    saveData(args.work_dir + fname, examples)
    return examples

'''
Generates example from sequence of words and preposition position
'''
def generateSamplesFromSent(sent, position, classes_to_id, words_vocabulary, tags_vocabulary, parse_vocabulary, args):
    example = dm.ExampleData()
    words = sent['words']
    tags = sent['tags']
    parse = sent['parse']
    sent_size = len(tags)
    for j in range(sent_size):
        tag = tags[j]
        if tag in args.not_include_classes:
            continue
        if j == position:
            example.pos_original = len(example.words)
            if args.original == "include":
                example.words.append([int(getIdFromWord(words_vocabulary.value_to_id, words[j]))])
            elif args.original == "replace":
                example.words.append([int(words_vocabulary.value_to_id['<tag_target_prep>'])])
            if args.original != "exclude":
                example.tags.append([int(tags_vocabulary.value_to_id[tags[j]])])
                example.parse.append([int(parse_vocabulary.value_to_id[parse[j]])])
            continue
        example.words.append([int(getIdFromWord(words_vocabulary.value_to_id, words[j]))])
        example.tags.append([int(tags_vocabulary.value_to_id[tags[j]])])
        example.parse.append([int(parse_vocabulary.value_to_id[parse[j]])])
    example.original = ([0]*len(classes_to_id))
    example.original[classes_to_id[words[position].lower()]] = 1
    example.trim(args)
    return example


def getGenerator(type_examples):
    return {
            'two_boundaries': 'generateSamplesFromSentTwoBoundaries',
            'one_boundary': 'generateSamplesFromSentOneBoundary',
            'tags_one_boundary': 'generateSamplesFromSentOneBoundaryAndTags',
            'full_sentence': 'generateSamplesFromSentFullSequenceAndTags',
            'full_sentence_window_2': 'generateSamplesFromSentOneBoundaryAndTags',
            'full_sentence_without_det_adj': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTags',
            'full_sentence_without_det_adj_balanced_data_': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTags',
            'full_lstm_tags': 'generateSamplesFromSentFullSequenceIncludingTags',
            'full_without_lstm_tags': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTags',
            'full_sentence_without_det_adj_balanced_data_with_tags': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTags',
            'full_without_lstm_tags_bi': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTags',
            'full_without_lstm_tags_bi_dropout': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTags',
            'full_without_lstm_tags_bi_dropout_attention': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTags',
            'two_ways': 'generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTagsTwoWays'
            }[type_examples]

'''
Two boundaries:
    full stop
    other preposition.
replaced by <tag_empty>
'''
def generateSamplesFromSentTwoBoundaries(sent, position, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id):
    ti = []
    to = []
    words = sent['words']
    tags = sent['tags']
    tag_size = len(tags)
    after_prep = 0
    before_prep = 0
    for j in range(1, 6):
        if position - j >= 0 and tags[position - j] != 'IN' and before_prep == 0:
            ti.insert(0, [int(vocabulary_to_id[words[position - j].lower()])])
        else:
            ti.insert(0, [int(vocabulary_to_id['<tag_empty>'])])
        if position - j >= 0 and tags[position - j] == 'IN':
            before_prep = 1
        if position + j < tag_size and tags[position + j] != 'IN' and after_prep == 0:
            ti.insert(len(ti), [int(vocabulary_to_id[words[position + j].lower()])])
        else:
            ti.insert(len(ti), [int(vocabulary_to_id['<tag_empty>'])])
        if position + j < tag_size and tags[position + j] == 'IN':
            after_prep = 1
    to = ([0]*10)
    to[classes_to_id[words[position]]] = 1
    return ti, to, [], [], []
'''
One boundary:
    full stop, replaced by <tag_empty>
Another preposition, replaced by <tag_another_prep>
'''
def generateSamplesFromSentOneBoundary(sent, position, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id):
    ti = []
    to = []
    words = sent['words']
    tags = sent['tags']
    tag_size = len(tags)
    for j in range(1, 6):
        if position - j >= 0 and tags[position - j] != 'IN':
            ti.insert(0, [int(vocabulary_to_id[words[position - j].lower()])])
        else:
            if position - j >= 0 and tags[position - j] == 'IN':
                ti.insert(0, [int(vocabulary_to_id['<tag_another_prep>'])])
            else:
                ti.insert(0, [int(vocabulary_to_id['<tag_empty>'])])
        if position + j < tag_size and tags[position + j] != 'IN':
            ti.insert(len(ti), [int(vocabulary_to_id[words[position + j].lower()])])
        else:
            if position + j < tag_size and tags[position + j] == 'IN':
                ti.insert(len(ti), [int(vocabulary_to_id['<tag_another_prep>'])])
            else:
                ti.insert(len(ti), [int(vocabulary_to_id['<tag_empty>'])])
    to = ([0]*10)
    to[classes_to_id[words[position]]] = 1
    return ti, to, [], [], []

'''
One boundary:
    full stop, replaced by <tag_empty>
Another preposition, replaced by <tag_another_prep>
'''
def generateSamplesFromSentFullSequenceAndTags(sent, position, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id, original=False, max_length = 10):
    ti = []
    to = []
    tgs = []
    final_position = 0
    words = sent['words']
    tags = sent['tags']
    sent_size = len(tags)
    for j in range(sent_size):
        if j == position:
            final_position = len(ti)
            if original:
                ti.append([int(getIdFromWord(vocabulary_to_id, words[j]))])
            else:
                ti.append( [int(vocabulary_to_id['<tag_target_prep>'])])
            tgs.append([int(tags_name_to_id[tags[j]])])
            continue
        ti.append([int(getIdFromWord(vocabulary_to_id, words[j]))])
        tgs.append([int(tags_name_to_id[tags[j]])])
    to = ([0]*10)
    to[classes_to_id[words[position]]] = 1
    return trim(ti, max_length, final_position), to, trim(tgs, max_length, final_position), [], []

'''
One boundary:
    full stop, replaced by <tag_empty>
Another preposition, replaced by <tag_another_prep>
'''
def generateSamplesFromSentFullSequenceIncludingTags(sent, position, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id, original= False, max_length = 10):
    ti = []
    to = []
    tgs = []
    final_position = 0
    words = sent['words']
    tags = sent['tags']
    sent_size = len(tags)
    for j in range(sent_size):
        tag = tags[j]
        if tag in ['DT', 'JJ']:
            continue
        if j == position:
            final_position = len(ti)
            if original:
                ti.append([int(getIdFromWord(vocabulary_to_id, words[j]))])
            else:
                ti.append([int(vocabulary_to_id['<tag_target_prep>'])])
            tgs.append([int(tags_name_to_id[tags[j]])])
            continue
        ti.append([int(getIdFromWord(vocabulary_to_id, words[j]))])
        tgs.append([int(tags_name_to_id[tags[j]])])
    to = ([0]*len(classes_to_id))
    to[classes_to_id[words[position].lower()]] = 1
    return trim(ti, max_length, final_position), to, trim(tgs, max_length, final_position), [], []

def getIdFromWord(vocabulary_to_id, word):
    word = word.lower()
    if word not in vocabulary_to_id:
        #print(word)
        return vocabulary_to_id['<tag_unknown>']
    else:
        return vocabulary_to_id[word]

'''
One boundary:
    full stop, replaced by <tag_empty>
Another preposition, replaced by <tag_another_prep>
'''
def generateSamplesFromSentFullSequenceWithoutDetAdjIncludingTagsTwoWays(sent, position, classes_to_id, top_classes, vocabulary_to_id, tags_name_to_id, original = False, max_length = 10):
    max_length = int(max_length/2)
    tifw = []
    tibw = []
    to = []
    tgsfw = []
    tgsbw = []
    words = sent['words']
    tags = sent['tags']
    sent_size = len(tags)
    #Forward up to prep
    for j in range(position):
        tag = tags[j]
        if tag in ['DT', 'JJ']:
            continue
        tifw.append([int(getIdFromWord(vocabulary_to_id, words[j]))])
        tgsfw.append([int(tags_name_to_id[tags[j]])])     

    for j in range(sent_size - 1, position, -1):
        tag = tags[j]
        if tag in ['DT', 'JJ']:
            continue
        tibw.append([int(getIdFromWord(vocabulary_to_id, words[j]))])
        tgsbw.append([int(tags_name_to_id[tags[j]])])
    if original:
        tifw.append([int(vocabulary_to_id[words[position].lower()])])
        tibw.append([int(vocabulary_to_id[words[position].lower()])])
    else:
        tifw.append([int(vocabulary_to_id['<tag_target_prep>'])])
        tibw.append([int(vocabulary_to_id['<tag_target_prep>'])])
    tgsfw.append([int(tags_name_to_id[tags[position]])])
    tgsbw.append([int(tags_name_to_id[tags[position]])])
    to = ([0]*10)
    to[classes_to_id[words[position]]] = 1
    return trim(tifw, max_length, len(tifw)), to, trim(tgsfw, max_length, len(tgsfw)), trim(tibw, max_length, len(tibw)), trim(tgsbw, max_length, len(tgsbw))

'''
Trim sequence according to length
'''
def trim(long_list, max_length, position):
    short_list = []
    if len(long_list) > max_length:
        #trim left
        start = position - int(max_length/2)
        if start < 0:
            start = 0
        for i in range(start, position):
            short_list.append(long_list[i])
        #trim right
        end = position + int(max_length/2)
        if end > len(long_list):
            end = len(long_list)
        for i in range(position, end):
            short_list.append(long_list[i])
    else:
        short_list = long_list
    return short_list

'''
All number of examples per class are balanced
A tenth class is introduced: other_prep (meaning for other preps)
'''
def balanceExamplesTwoWays(examples_fw, examples_bw, original_label, corrected_label, tags_fw, tags_bw, parse_fw, parse_bw, classes, class_limit):
    EXAMPLES_SIZE = len(examples_fw)
    class_limit_size = 5
    #we take ten most common preps
    max_size_per_class = classes[class_limit_size][1]
    counter = [0]*class_limit

    rand_ids = [x for x in range(EXAMPLES_SIZE)]
    shuffle(rand_ids)

    original_label_temp = []
    corrected_label_temp = []
    examples_temp_fw = []
    tags_temp_fw = []
    examples_temp_bw = []
    tags_temp_bw = []
    parse_temp_fw = []
    parse_temp_bw = []

    for x in range(0, EXAMPLES_SIZE):
        current_class = np.argmax(corrected_label[rand_ids[x]])
        #All other classes are grouped into tenth class
        if current_class >= class_limit:
            continue
            #current_class = class_limit + 1
        target_prep = class_limit * [0]
        target_prep[current_class] = 1
        #print(target_prep)
        #print(target_prep)
        if counter[current_class] <= max_size_per_class:
            original_label_temp.append(original_label[rand_ids[x]])
            corrected_label_temp.append(target_prep)
            examples_temp_fw.append(examples_fw[rand_ids[x]])
            tags_temp_fw.append(tags_fw[rand_ids[x]])
            parse_temp_fw.append(parse_fw[rand_ids[x]])
            if len(examples_bw) > 0:
                examples_temp_bw.append(examples_bw[rand_ids[x]])
                tags_temp_bw.append(tags_bw[rand_ids[x]])
                parse_temp_bw.append(parse_bw[rand_ids[x]])
            counter[current_class] = counter[current_class] + 1
            
    return examples_temp_fw, examples_temp_bw, original_label_temp, corrected_label_temp, tags_temp_fw, tags_temp_bw, parse_temp_fw, parse_temp_bw


'''
All number of examples per class are balanced
A tenth class is introduced: other_prep (meaning for other preps)
'''


def shuffleData(examples_fw, examples_bw, original_label, corrected_label, tags_fw, tags_bw):
    EXAMPLES_SIZE = len(examples_fw)

    rand_ids = [x for x in range(EXAMPLES_SIZE)]
    shuffle(rand_ids)

    original_label_temp = []
    corrected_label_temp = []
    examples_temp_fw = []
    tags_temp_fw = []
    examples_temp_bw = []
    tags_temp_bw = []

    for x in range(0, EXAMPLES_SIZE):
        original_label_temp.append(original_label[rand_ids[x]])
        corrected_label_temp.append(corrected_label[rand_ids[x]])
        examples_temp_fw.append(examples_fw[rand_ids[x]])
        tags_temp_fw.append(tags_fw[rand_ids[x]])
        if len(examples_bw) > 0:
            examples_temp_bw.append(examples_bw[rand_ids[x]])
            tags_temp_bw.append(tags_bw[rand_ids[x]])

    return examples_temp_fw, examples_temp_bw, original_label_temp, corrected_label_temp, tags_temp_fw, tags_temp_bw
            
def balanceExamples(examples_fw, original_label, corrected_label, tags_fw, classes):
    EXAMPLES_SIZE = len(examples_fw)
    class_limit = 8
    #we take ten most common preps
    max_size_per_class = classes[class_limit][1]
    counter = [0]*10

    rand_ids = [x for x in range(EXAMPLES_SIZE)]
    shuffle(rand_ids)

    original_label_temp = []
    corrected_label_temp = []
    examples_temp_fw = []
    tags_temp_fw = []

    for x in range(0, EXAMPLES_SIZE):
        current_class = np.argmax(corrected_label[rand_ids[x]])
        #All other classes are grouped into tenth class
        if current_class > class_limit:
            current_class = class_limit + 1
        if counter[current_class] <= max_size_per_class:
            original_label_temp.append(original_label[rand_ids[x]])
            corrected_label_temp.append(corrected_label[rand_ids[x]])
            examples_temp_fw.append(examples_fw[rand_ids[x]])
            tags_temp_fw.append(tags_fw[rand_ids[x]])
            counter[current_class] = counter[current_class] + 1

    return examples_temp_fw, original_label_temp, corrected_label_temp, tags_temp_fw
'''
Devide examples in a balanced way
'''
def devideBalancedExamples(training_examples_fw, training_student_preps, training_correct_preps, training_tags_fw):
    EXAMPLES_SIZE = len(training_examples_fw)
    length_training_examples = int(0.9 * EXAMPLES_SIZE)
    length_validation_examples = EXAMPLES_SIZE - length_training_examples

    max_validation_size_per_class = length_validation_examples // 10
    counter = [0]*10

    rand_ids = [x for x in range(EXAMPLES_SIZE)]
    shuffle(rand_ids)
    #Training
    original_label_training = []
    corrected_label_training = []
    examples_training_fw = []
    tags_training_fw = []
    #Validation
    original_label_validation = []
    corrected_label_validation = []
    examples_validation_fw = []
    tags_validation_fw = []

    for x in range(0, EXAMPLES_SIZE):
        current_class = np.argmax(training_correct_preps[rand_ids[x]])
        if counter[current_class] <= max_validation_size_per_class:
            original_label_validation.append(training_student_preps[rand_ids[x]])
            corrected_label_validation.append(training_correct_preps[rand_ids[x]])
            examples_validation_fw.append(training_examples_fw[rand_ids[x]])
            tags_validation_fw.append(training_tags_fw[rand_ids[x]])
            counter[current_class] = counter[current_class] + 1
        else:
            original_label_training.append(training_student_preps[rand_ids[x]])
            corrected_label_training.append(training_correct_preps[rand_ids[x]])
            examples_training_fw.append(training_examples_fw[rand_ids[x]])
            tags_training_fw.append(training_tags_fw[rand_ids[x]])  
        
    return examples_training_fw, original_label_training, corrected_label_training, tags_training_fw, examples_validation_fw, original_label_validation, corrected_label_validation, tags_validation_fw

def divideExamples(examples, args):
    fname_training = createNameForExamples("data/training_divided_examples_", args)
    fname_validation = createNameForExamples("data/validation_divided_examples_", args)
    saved_examples = restoreObject(fname_training, args)
    if saved_examples != 0:
        return saved_examples, restoreObject(fname_validation, args)

    EXAMPLES_SIZE = len(examples)
    #Get number examples per class
    counter = {}
    for example in examples:
        class_example = np.argmax(example.correct)
        if class_example not in counter:
            counter[class_example] = 1
        else:
            counter[class_example] = counter[class_example] + 1

    #Set max value for small dataset (10% per class)
    max_per_class = args.class_limit * [0]
    for i in range(args.class_limit):
        max_per_class[i] = counter[i] * 0.1
    #Shuffle examples
    rand_ids = [x for x in range(EXAMPLES_SIZE)]
    shuffle(rand_ids)
    #Training
    training_examples = []
    #Validation
    validation_examples = []

    # devide examples
    counter = [0]*args.class_limit
    for x in range(0, EXAMPLES_SIZE):
        current_class = np.argmax(examples[rand_ids[x]].correct)
        if counter[current_class] <= max_per_class[current_class]:
            validation_examples.append(examples[rand_ids[x]])
            counter[current_class] = counter[current_class] + 1
        else:
            training_examples.append(examples[rand_ids[x]])
    saveData(args.work_dir + fname_training, training_examples)
    saveData(args.work_dir + fname_validation, validation_examples)
    return training_examples, validation_examples
'''
Return undersampling
'''
def underSample(examples, args):
    EXAMPLES_SIZE = len(examples)
    under_examples = []
    counter = [0] * args.class_limit
    for x in range(EXAMPLES_SIZE):
        current_class = np.argmax(examples[x].correct)
        counter[current_class] = counter[current_class] + 1

    # undersample
    max_num_examples = counter[args.under_pos]
    counter = [0] * args.class_limit
    for x in range(EXAMPLES_SIZE):
        current_class = np.argmax(examples[x].correct)
        if counter[current_class] <= max_num_examples:
            under_examples.append(examples[x])
            counter[current_class] = counter[current_class] + 1
    return under_examples
'''
Return oversampling
'''
def overSample(examples, args):
    training_per_classes = group_by_class(examples)
    over_examples = []
    counter_max = [0] * args.class_limit
    for x in range(args.class_limit):
        counter_max[x] = len(training_per_classes[x]) - 1

    # oversample
    mim_num_examples = counter_max[args.over_pos]
    counter = [0] * args.class_limit
    for j in range(args.class_limit):
        for i in range(counter_max[0]):
            if counter[j] < counter_max[j]:
                over_examples.append(training_per_classes[j][i])
            elif j > args.over_pos and counter[j] < mim_num_examples:
                ran_ind = random.randint(0, counter_max[j])
                over_examples.append(training_per_classes[j][ran_ind])
            else:
                break
            counter[j] += 1
    # randomize
    rand_ids = [x for x in range(len(over_examples))]
    shuffle(rand_ids)

    random_over_examples = []
    for id in rand_ids:
        random_over_examples.append(over_examples[id])

    return random_over_examples

'''
Returns probabilities as one hot vectors
'''
def get_one_hot_from_prob(prep_prob_test, args):
    index_prep_test = [np.argmax(p) for p in prep_prob_test]
    corrector_prep = []
    for i in index_prep_test:
        temp = [0]*args.class_limit
        temp[i] = 1
        corrector_prep.append(temp)
    return corrector_prep

def accuracy(predictions, labels):
    truths = 0
    for i in range(predictions.shape[0]):
        #print("Prediction: ", np.argmax(predictions[i]))
        #print("Real value: ", np.argmax(labels[i]))
        if(np.argmax(predictions[i]) == np.argmax(labels[i])):
            truths = truths + 1
    return (truths/predictions.shape[0])

'''
Remove sentences longer than argument 'max_length'
'''
def removeExamplesOutsideTrainingTarget(examples, args):
    fname_training = createNameForExamples("data/training_removed_examples_", args)
    saved_examples = restoreObject(fname_training, args)
    if saved_examples != 0:
        return saved_examples

    examples_temp = []
    EXAMPLES_SIZE = len(examples)

    #remove examples
    for x in range(0, EXAMPLES_SIZE):
        current_class = np.argmax(examples[x].correct)
        if current_class < args.class_limit:
            correct = [0] * args.class_limit
            correct[current_class] = 1
            examples[x].correct = correct
            examples_temp.append(examples[x])
    saveData(args.work_dir + fname_training, examples_temp)
    return examples_temp

def group_by_class(training_examples):
    training_by_classes = {}
    for example in training_examples:
        current_class = np.argmax(example.correct)
        if current_class not in training_by_classes:
            training_by_classes[current_class] = []
        training_by_classes[current_class].append(example)
    return training_by_classes

def createName(prefix, args):
    name = str(prefix +
                "_Type_" + args.model_type +
                "_Data_" + args.type_data +
                "_D_" + str(args.dropout) +
                "K_" + str(args.keep_prob).replace(".", "") +
                "_A_" + str(args.attention) +
                "_Spell_" + str(args.spelling) +
                "_Ori_" + str(args.original) +
                "_L_" + str(args.class_limit) +
                "_Hnum_" + str(args.num_hidden) +
                "_Lnum_" + str(args.num_layer) +
                "_Tags_" + str(args.tags) +
                str(args.tags_emb_size) +
                "_Parse_" + str(args.parse) +
                str(args.index_emb_size) +
                "_Under_" + str(args.under) +
                "_N_" + str(args.under_pos) +
                "_Over_" + str(args.over) +
                "_N_" + str(args.over_pos) +
                "_I_" + str(args.type_input) +
                "_Win_" + str(args.size_window) +
                "_Sk_" + str(args.skip_num) +
                "_Sa_" + str(args.num_sampled) +
                "_Sq_" + str(args.seq_limit)
               #+ "_L" + str(args.lower_limit)
               #+ "E" + str(args.word_emb_size)

            )
    if args.original != 'exclude':
        name = name + "_L" + str(args.lower_limit) + "E" + str(args.word_emb_size)
    if 'predictor' in prefix:
        name = name + "_Mode_train"
    else:
        name = name + "_Mode_" + str(args.mode)
    return name

def createNameForExamples(prefix, args):
    name = str(prefix +
                "_Data_" + args.type_data +
                "_Spell_" + str(args.spelling) +
                "_Ori_" + str(args.original) +
                "_L_" + str(args.class_limit) +
                "_Under_" + str(args.under) +
                "_N_" + str(args.under_pos) +
                "_Over_" + str(args.over) +
                "_N_" + str(args.over_pos) +
                "_Sq_" + str(args.seq_limit) +
                "_LL" + str(args.lower_limit)
            )
    return name