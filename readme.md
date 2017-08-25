**Last updated on August 24th, 2017** by [Mirith](https://github.com/Mirith)

# Overview

This project uses many modules from [nltk](http://www.nltk.org/api/nltk.html).  And [mean](https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html) from numpy to average the results.  It takes an previously tagged dataset and trains a trigram tagger based on that dataset.  The trigram tagger backs off into a bigram tagger, which backs off into a unigram tagger, which in turn backs off into a default tagger.  

# Usage

You'll need python (this was done in python 3) and the dataset.  The small one provided is probably not going to give you very accurate results (with the full set it's about 88-89% accurate).  But it will give you an idea of how it works, while drastically reducing the training time.  

# Files

## estonianSmall.txt

Tagged Estonian data.  Each word has its own tag, comprised of one letter.  IE

> *word/single letter tag*

Only includes the first 200 lines of a much, much larger dataset.  

## taggers.py

Currently kfold validation doesn't work unless hard-coded, which is less than ideal.  But this file basically loads the tagged corpus, then splits the data, then trains based off the hard-coded split data, and prints the results.  Capitalizing all the words improves accuracy just slightly.  

