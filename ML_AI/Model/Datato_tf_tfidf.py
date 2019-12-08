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


## -- Get the Data from ArXiv
def main(argv):
    if len(argv)!=0:
        print("Input: ", argv)
        raise Exception("USE: python Datato_tf_tfidf.py")

    data_dfKeys = ['id','authors','published','title','abstitle',"prim_cat"]
    data_df = pd.DataFrame(columns=data_dfKeys)

    tmp_dict = {}
    for year in range(1993,2020):
        filename = "../Arxiv_DataSet/ArXivCS_"+str(year)+".pickle"
        tmp_df = pd.read_pickle(filename)
        tmp_dict[year] = len(tmp_df)
        #print year, len(tmp_df)
        data_df = data_df.append(tmp_df, ignore_index=True)
    # del tmp_dict[2017]

    # -- Dropping duplicates!
    print("duplicated entries = ", data_df['id'].duplicated().sum())
    data_df = data_df.drop_duplicates(subset="id")
    data_df = data_df.reset_index(drop=True)
    print("duplicated entries after removal = ", data_df['id'].duplicated().sum())
    print("total number of documents: ", len(data_df))

    # see the data-frame by sorted by arxiv-id
    # data_df.sort_values(by="id").head()

    # -- creating primary category, word_count of each abstract, and word_count of each title - as potential features
    data_df["prim_cat"] = data_df["prim_cat"].apply(fix_cat);
    data_df["word_count"] = data_df["abstitle"].apply(length_of);
    data_df["title_count"] = data_df["title"].apply(length_of);

    ## -- clean text, lowercase, remove stop words, stem and lemmatize the words --
    print("applying all the cleaning functions")
    data_df["cleaned_abstitle"] = data_df["abstitle"].apply(apply_all)

    ## -- dump cleaned data to the csv file.
    data_df.to_csv("cleaned_data.csv")
    # data_df = pd.read_csv("cleaned_data.csv")

    # -- Printing something - just to check if things are done properly!
    print(" \n", "check if all the cleaning is done properly")
    print(data_df.iloc[1]["abstitle"], " \n")
    print(data_df.iloc[1]["cleaned_abstitle"])

    # -- <h2> Preparing Data for LDA </h2> - tokonize all text
    text_data = []
    for line in data_df["cleaned_abstitle"]:
        tokens = line.split(" ");
        text_data.append(tokens)

    # -- creating dictionary of all unique words, and copus - bag of word for each document
    dictionary = corpora.Dictionary(text_data)

    # -- tf vectorize
    print("term-freq calculations")
    tf_corpus = [dictionary.doc2bow(text) for text in text_data]

    # -- term frequency inverse document frequency (tfidf) - vectorize
    print("tf-idf calculations")
    tfidf_model = TfidfModel(tf_corpus)  # fit model
    tfidf_corpus = tfidf_model[tf_corpus]      # apply model to the first corpus document

    # -- Let's print out tf and tfidf of a document - just as a check.
    print(" \n", "another check after tf/tfidf calculations")
    print(data_df.iloc[1]["cleaned_abstitle"])
    for i in range(0,len(tf_corpus[1])):
        word_index = tf_corpus[1][i][0]
        print(tf_corpus[1][i], tfidf_corpus[1][i], dictionary[word_index])

    # -- printing corpus and dictionary to a file
    pickle.dump(tf_corpus, open('corpus.pkl', 'wb'))
    pickle.dump(tfidf_corpus,open("tfidf_corpus.pkl","wb"))
    pickle.dump(tfidf_model,open("tfidfmodel.pkl","wb"))

    # -- dictionary.save('dictionary.gensim')
    pickle.dump(dictionary,open("dictionary.pkl","wb"))
    pickle.dump(text_data, open("text_tokonized.pkl", "wb"))

    # -- reading in all the printed things!
    # tf_corpus = pickle.load(open("corpus.pkl","rb"))
    # tfidf_corpus = pickle.load(open("tfidf_corpus.pkl","rb"))
    # tfidf_model = pickle.load(open("tfidfmodel.pkl","rb"))
    # dictionary = pickle.load(open("dictionary.pkl","rb"))




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

