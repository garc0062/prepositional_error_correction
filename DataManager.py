# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 13:59:17 2018

@author: pablo
"""
import DataProcessor as dp
import numpy as np

class ExampleData:
    def __init__(self):
        self.original = []
        self.correct = []
        self.predicted = []
        self.tags = []
        self.parse = []
        self.words = []
        self.tags_bw = []
        self.parse_bw = []
        self.words_bw = []
        self.pos_original = 0

    def length(self):
        return len(self.words)

    def getEvaluation(self):
        result = "Correct: " + str(np.argmax(self.correct)) + " Original: " + str(np.argmax(self.original)) + \
                 " Predicted: " + str(np.argmax(self.predicted)) + "\t\n"
        return result


    def trim(self, args):
        max_length = 2 * args.seq_limit + 1
        self.tags = dp.trim(self.tags, max_length, self.pos_original)
        self.parse = dp.trim(self.parse, max_length, self.pos_original)
        self.words = dp.trim(self.words, max_length, self.pos_original)
        self.tags_bw = dp.trim(self.tags_bw, max_length, self.pos_original)
        self.parse_bw = dp.trim(self.parse_bw, max_length, self.pos_original)
        self.words_bw = dp.trim(self.words_bw, max_length, self.pos_original)
        start = self.pos_original - args.seq_limit
        if start < 0:
            start = 0
        self.pos_original = self.pos_original - start

class Vocabulary:
    def __init__(self, value_to_id, reverse):
        self.value_to_id = value_to_id
        self.reverse = reverse

    def length(self):
        return len(self.value_to_id)

def getSizeInput(args):
    size_input = args.word_emb_size
    if args.tags:
        size_input = size_input + args.tags_emb_size
    if args.parse:
        size_input = size_input + args.index_emb_size
    return size_input