#!/usr/bin/env python
# coding: utf-8

# Import all libraries
import urllib, string, os, math, cmath, re
import sys, time, matplotlib, pickle
import numpy as np
import pandas as pd
import _pickle as cPickle
from time import time
import matplotlib.pyplot as plt

# scikit-learn packages
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import shuffle
from scipy.stats import entropy

# nltk and gensim packages
import nltk, gensim
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist
from gensim import corpora
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import models, similarities
from gensim.models import CoherenceModel


## -- Get the Data from ArXiv
def main(argv):
    if len(argv)!=0:
        print("Input: ", argv)
        raise Exception("USE: python Datato_tf_tfidf.py")

    # -- reading in all the printed things!
    tf_corpus = pickle.load(open("../corpus.pkl","rb"))
    tfidf_corpus = pickle.load(open("../tfidf_corpus.pkl","rb"))
    tfidf_model = pickle.load(open("../tfidfmodel.pkl","rb"))
    dictionary = pickle.load(open("../dictionary.pkl","rb"))
    text_data = pickle.load(open("../text_tokonized.pkl", "rb"))


    """
    <h1> LDA modeling </h1>
    
    Documentation: 
    https://radimrehurek.com/gensim/models/ldamodel.html <br>
    class gensim.models.ldamodel.LdaModel(corpus=None, num_topics=100, id2word=None, 
                                          distributed=False, chunksize=2000, passes=1, 
                                          update_every=1, alpha='symmetric', eta=None, 
                                          decay=0.5, offset=1.0, eval_every=10, iterations=50, 
                                          gamma_threshold=0.001, minimum_probability=0.01, 
                                          random_state=None, ns_conf=None, minimum_phi_value=0.01, 
                                          per_word_topics=False, callbacks=None, 
                                          dtype=<class 'numpy.float32'>)
    
    """
    out = []
    for Ntopics in range(30,60,1):

        # training model
        t0 = time()
        ldamodel = gensim.models.ldamulticore.LdaMulticore(
            tfidf_corpus, num_topics = Ntopics, chunksize=10000,
            id2word=dictionary, iterations=1000, passes=40,
            workers=4, random_state=1)

        time_LDA = time() - t0

        # save the model!
        fname = "tfidf_model"+str(Ntopics)+".gensim"
        ldamodel.save(fname)

        perplexity = ldamodel.log_perplexity(tfidf_corpus)
        umass_coherencemodel = CoherenceModel(model=ldamodel, corpus=tfidf_corpus,
                                              dictionary=dictionary, coherence='u_mass')
        umass_coherence_value = umass_coherencemodel.get_coherence()

        cv_coherencemodel = CoherenceModel(model=ldamodel, corpus=tfidf_corpus, texts=text_data,
                                           dictionary=dictionary, coherence='c_v')
        cv_coherence_value = cv_coherencemodel.get_coherence()

        print(Ntopics, perplexity, umass_coherence_value, cv_coherence_value, time_LDA)
        out.append([Ntopics, perplexity, umass_coherence_value, cv_coherence_value, time_LDA])


    out = np.asarray(out)
    np.savetxt("models_summary.csv", out, delimiter=",")


#######################################################################################################################
#######################################################################################################################

# Fix category labels - needed for only condensed matter categories!
def fix_cat(in_cat): 
    if(in_cat=="cond-mat.stat-mech"):
        return "stat-mech"
    if(in_cat=="cond-mat.supr-con"):
        return "supr-con"
    if(in_cat=="cond-mat.dis-nn"):
        return "dis-nn"
    if(in_cat=="cond-mat.mes-hall"):
        return "mes-hall"
    if(in_cat=="cond-mat.str-el"):
        return "str-el"
    if(in_cat=="cond-mat.mtrl-sci"):
        return "mtrl-sci"
    if(in_cat=="cond-mat.soft"):
        return "soft"
    if(in_cat=="cond-mat.quant-gas"):
        return "quant-gas"
    if(in_cat=="cond-mat.other"):
        return "other"
    if(in_cat=="cond-mat"):
        return "other"
    return in_cat; 
    

# returns length of the input text
def length_of(input_val): 
    return len(input_val);


# Clean input using basic things.
def initial_clean(text):

    # remove all punctuations and links and strange symbols
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)

    # lower case all text
    text = text.lower()

    # removed some very common words - model, predict, algorithm, learn
    #text = text.replace("model ","");
    #text = text.replace("predict ","");
    #text = text.replace("algorithm ","");
    #text = text.replace("learn ","");

    # tokonize words for further cleaning
    # text = text.split(",")
    text = nltk.word_tokenize(text)

    return text

# Removes all stopwords from text
stop_words = stopwords.words('english')
def remove_stop_words(text):
    return [word for word in text if word not in stop_words]

# Stem/Lemmatize words of input text
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
def stem_words(text):
    try:
        text = [stemmer.stem(word) for word in text]
        text = [lemmatizer.lemmatize(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

# Apply all the above functions - initial_clean, remove_stop_words, stem_words
def apply_all(text):

    # text = text.encode("utf8")
    
    words = stem_words(remove_stop_words(initial_clean(text)))
    outtext = " ".join([x for x in words])

    return outtext






# main file - input
if __name__ == "__main__":
    main(sys.argv[1:])

