#word tokenizer = separate words for us
#word lemmatizer = takes similar word and converts them into the same single word

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import random #shuffle data
import pickle # save the process
from collections import Counter  #sort most common lemmas
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
hm_lines = 100000

#plan: create a bernoulli bag of word model
def create_lexicon(pos, neg):

    lexicon = []
    with open(pos, 'r') as f:
        contents = f.readlines()
        #iterate through every line
        for l in contents[:hm_lines]:
            #tokenize line
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    with open(neg, 'r') as f:

        contents = f.readlines()
        # iterate through every line
        for l in contents[:hm_lines]:
            # tokenize line
            all_words = word_tokenize(l)
            lexicon += list(all_words)

    #want to remove duplicates
    #remove common word - stopwords - i.e. a, and, or etc
    #also remove super uncommon words

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        #print(w_counts[w]) -> dictionary hence each word associated to a value
        if 1000 > w_counts[w] > 50:
            #usually 1000 and 50 would kind of just be a % of the entire dataset
            #we are adding to our lexicon words that occur within that range
            l2.append(w)

    print(len(l2))
    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        f.decode("utf8")
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            #create a numpy arrays of zeros
            #which is the lengt of our lexicon
            features = np.zeros(len(lexicon))

            for word in current_words:
                #if lemmatized word is within our lexicon
                if word.lower() in lexicon:
                    #get index fo that word and update the bernoulli array
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features,classification])

    return  featureset

#training set of features
def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos,neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0]) #[1,0] is the one_hot vector classifying the thing as positive
    features += sample_handling('neg.txt', lexicon, [0,1]) #[0,1] one_hot for the negatvie classification
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))


    #this is creating the datasets that will then be used for the trainign/testing

    #traing x = the features
    #traing y = the labels of the features in train x
    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y


if __name__ == 'main':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    #if you want to pickle the data
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x,train_y,test_x,test_y], f)



