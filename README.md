# IberEval2017
This code is developed to take part into IberEval2017 competition .

In particular Classification Of Spanish Election Tweets (COSET) and  STANCE AND GENDER DETECTION IN TWEETS ON CATALAN

Task deadlines:
March 20th, 2017 Release of training data. 

April 24th, 2017 Release of test data.

May 08th, 2017 Submission of runs.

May 15th, 2017 Evaluation results.

May 29th, 2017 Working notes due. 

June 12nd, 2017 Review to authors. 

June 26th, 2017 Camera ready due.


Contacts:

Mail: coset2017@gmail.com

Mailing list: coset@googlegroups.com

Developers:
Ambrosini Luca - luca.ambrosini@supsi.ch

Giancarlo Nicolò - giani1@inf.upv.es

Used wordvectors: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.es.vec

# Tweet representations:

- Bag of words
    - tf-idf normalization
    
- Bag of n-grams
    - tf-idf normalization
    
- Word embeddings
    - Leaned online
    - fasttext es improved online
    - fasttext es static
    - fasttext ca improved online
    - fasttext ca static
- N-grams

# Classifiers

- Random forest
- Support Vector Machines
- Decision trees

# Deep neural models

- Multi Layer Perceptron
- CNN
- LSTM
- CNN+LSTM
- BI-LSTM
- KIM
- FAST-TEXT

# Pre-processing

- Stemming
- Remove stop-words
- Clean url
- Clean numbers
- Clean twitter reserved words
- Tokenize mentions
- Tokenize Smiley
- Tokenize Emojies
