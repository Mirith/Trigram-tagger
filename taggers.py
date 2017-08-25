# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:34:41 2017

@author: Mirith

Info: HW1

"""

from nltk.corpus.reader import TaggedCorpusReader
from nltk import DefaultTagger, UnigramTagger, BigramTagger, TrigramTagger
from nltk.probability import FreqDist
from numpy import mean
# for kfold validation, not working though
# cross-fold validation is just brute forced...
#from sklearn.model_selection import KFold
#import numpy as np


mypath = "C:/Users/Lauren Shin/Documents/LING 111/.final project"

EstonianCorpus = TaggedCorpusReader(mypath, "estonianCaps.txt", encoding = "latin-1")

sentences = EstonianCorpus.tagged_sents()

tags = [tag for _, tag in EstonianCorpus.tagged_words()]
mostFrequent = FreqDist(tags).max()

default = DefaultTagger(mostFrequent)

# cross validation

#kf = KFold(n_splits = 3)
#
## turns the data into a 2d array
#X = np.array(sentences)
## creates a 1d array with same length/number of rows as X
#y = np.arange(0, len(sentences), 1)
#
#for train, test in kf.split(X):
#    # this works
#    # training for training and training for evaluation
#    X_train, X_test = X[train], X[test]
#    # testing for training and testing for evaluation
#    y_train, y_test = y[train], y[test]
#    print(train, test)
    
    # this does not work
    # Unigram = UnigramTagger(X_train, backoff = default)
    # # throws 'ValueError: The truth value of an array with more than 
    # # one element is ambiguous. Use a.any() or a.all()'
#    Unigram.evaluate(y_train)
#    Bigram = BigramTagger(training, backoff = Unigram)
#    Trigram = TrigramTagger(training, backoff = Bigram)
#    print(Trigram.evaluate(testing))


    # brute forcing 3 fold validation... works, ~ 89% accurate trigram models

results = []
    
third = int(len(sentences) / 3)

chunk1 = sentences[:third]
chunk2 = sentences[third:-third]
chunk3 = sentences[-third:]

chunks = [chunk1, chunk2, chunk3]

def TrainTaggers(training, testing):
    global results
    Unigram = UnigramTagger(training, backoff = default)
    print('unigram trained')
    Bigram = BigramTagger(training, backoff = Unigram)
    print('bigram trained')
    Trigram = TrigramTagger(training, backoff = Bigram)
    print('trigram trained')
    results += [Trigram.evaluate(testing)]

# first chunk
temp = chunk1 + chunk2

TrainTaggers(temp, chunk3)

# second chunk
temp = chunk1 + chunk3

TrainTaggers(temp, chunk2)

# third chunk
temp = chunk3 + chunk2

TrainTaggers(temp, chunk1)

print(results)
# just ngrams
# [0.8827753007885345, 0.8951900637025421, 0.896695826872251]
# with all caps
# [0.8906369806605515, 0.9014164884263662, 0.90380545559164]

# averaging performance of taggers
performance = mean(results)
print(performance)
# just ngrams
# 0.891553730454
# with all caps
# 0.89861964156