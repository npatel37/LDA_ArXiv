import urllib
import string
import numpy as np
import pandas as pd
import cPickle
from time import time
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def main():
	data_dfKeys = ['id','authors','published','title','abstitle',"prim_cat"]
	data_df = pd.DataFrame(columns=data_dfKeys)
	data_df = shuffle(data_df)

	tmp_dict = {}
	for year in range(1992,2018): #range(1992,2018):
	    filename = "Arxiv_DataSet/ArXivCondMat_"+str(year)+".pickle"
	    tmp_df = pd.read_pickle(filename)
	    tmp_dict[year] = len(tmp_df)
	    print "year = ", year, " and  # doc = ", len(tmp_df)
	    data_df = data_df.append(tmp_df, ignore_index=True)
	del tmp_dict[2017]

	print "Pandas DataSet Keys: ", data_df.keys()
	print "total number of documents: ", len(data_df)
	print "Number of categories on ArXiv: ", len(data_df["prim_cat"].unique())
	print "Categories on ArXiv: ", data_df["prim_cat"].unique()

	## term frequency inverse document freq. (tf-idf)
	print "Building tfidf sparse matrix"
	tfidf_vectorizer = TfidfVectorizer()
	tfidf = tfidf_vectorizer.fit_transform(data_df['abstitle'])
	tfidf_feature_names = tfidf_vectorizer.get_feature_names()
	tfidf = normalize(tfidf)
	cPickle.dump(tfidf, open('model_parameters/tfidf.p', 'wb')) 
	cPickle.dump(tfidf_feature_names, open('model_parameters/tfidf_feature_names.p', 'wb'))

	## term frequency
	print "Building term frequency"
	tf_vectorizer = CountVectorizer()
	tf = tf_vectorizer.fit_transform(data_df['abstitle'])
	cPickle.dump(tf, open('model_parameters/tf.p', 'wb'))

	
	t0 = time()
	print "Starting to fit the LDA model @ time = ", t0
	lda = LatentDirichletAllocation(n_topics=25,
		                        max_iter=20,
		                        batch_size=128,
		                        learning_decay=0.7,
		                        learning_method='online',
		                        learning_offset=10.,
		                        mean_change_tol=0.001,
		                        random_state=None)

	lda.fit(tf)
	print("done in %0.3fs." % (time() - t0))

	cPickle.dump(lda, open('model_parameters/LDAclassifier.model', 'wb') )



if __name__ == "__main__":
	main()
