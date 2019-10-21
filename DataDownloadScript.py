import urllib
import string
import sys, os
import numpy as np
import pandas as pd
import _pickle as cPickle
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords') ### <------ this must be downloaded once!

def main(argv):
	try:
		year = argv[0]
	except Exception as e:
		print(e)
	#year = "2015"
	print(" Getting Data from ArXiv API Year = ", year)
	os.system("mkdir -p Arxiv_DataSet")
	monthsval = ['01','02','03','04','05','06','07','08','09','10','11','12']
	print(" ======================== \n")


	print("getting CondMat data")
	Data = [] 
	print("   Events length: ",end=" ")
	for month in monthsval:
		infile = "WebSources/Source_"+str(year)+"_"+str(month)+".txt"
		source = cPickle.load(open(infile, 'rb'),encoding='iso-8859-1')#.decode('utf8')
		events = source.split('<entry>')
		Data.append(events)
		print(len(source),end=",")

	print(" \n   Clean and add to pandas DataFrame")
	dfKeys = ['id','authors','published','title','abstitle',"prim_cat"]
	df = toDataFrame(Data,"cond-mat")
	Outfile2="Arxiv_DataSet/ArXivCondMat_"+str(year)+".pickle"
	df.to_pickle(Outfile2)
	del df, Data
	print(" ======================== ")



def toDataFrame(Data,catprefix):
	dfKeys = ['id','authors','published','title','abstitle',"prim_cat"]
	print("   Monthly Datasize: ",end=" ")
	df = pd.DataFrame(columns=dfKeys)
	for i, monthlyData in enumerate(Data):
		pandaData = webScrapArxiv(monthlyData,catprefix)
		month = "{0:0=2d}".format(i+1)
		df = df.append(pandaData, ignore_index=True)
	print(" \n ")
	return df 

def montlyData(urlmid):
	url_start = "http://export.arxiv.org/api/query?search_query=cat:cond-mat*+AND+submittedDate:["
	url_end = "9999]&start=0&max_results=10000"
	print("   ", url_start+urlmid+url_end)
	data = urllib.urlopen(url_start+urlmid+url_end).read()
	events = data.split('<entry>')
	print("   Events length: ", len(events))
	return events

def removePunctuations(s):
	s = s.replace("\n"," ")
	for c in string.punctuation:
		s=s.replace(c,"")
	return s.lower()

def webScrapArxiv(events,catprefix):
	dfKeys = ['id','authors','published','title','abstitle',"prim_cat"]
	dfmonth = pd.DataFrame(columns=dfKeys)
	for source in events[1:]:
		title = removePunctuations(source.split('<title>')[1].split('</title>')[0])
		abstract = removePunctuations(source.split('<summary')[1].split("</summary>")[0])
		abst_id = source.split('<id>http://arxiv.org/abs/')[1].split("</id>")[0]

		#print(title, abst_id)
#		try:
#			prim_cat = source.split('cond-mat.')[1].split("scheme=")[0].replace("\"","")
#		except Exception as e:
#			prim_cat = 'none'
#			#~ print "ArxivID: ", abst_id, source.split('cond-mat.'), e
#			#~ print " \n"

		#print("\n")
		prim_cat = source.split('<arxiv:primary_category')[1].split(" scheme=")[0].split(" term=")[1].replace("\"","")
		
		if(prim_cat.split(".")[0]!=catprefix):
			#print(prim_cat.split(".")[0], catprefix)
			continue; 


#		if(len(list(prim_cat))>10):
#			prim_cat = prim_cat.split("]")[0]
#			if(len(list(prim_cat))>11):
#				continue
#			prim_cat = prim_cat.replace(" ","")

		try:
			abst_id = abst_id.split('v')[0]
		except Exceptions as e:
			pass
		yrmonth = abst_id.split('.')[0]

		## -- Authors ----
		authors = source.split('<name>')
		auth_str = ""
		for i in range(1,len(authors)):
			names = authors[i]
			auth_str += names.split('</name>')[0]
			if (i<len(authors)-1):
				auth_str += ", "
				#auth_str = auth_str.replace(" ", "")
				#auth_str2 = (auth_str.replace(" ", "")).replace(",", " ").lower()

		# -- published or not ---
		try:
			pubtest = source.split('<arxiv:journal_ref')[1].split('</arxiv:journal_ref>')[0].split('atom">')[1]
		except Exception as e1:
			try:
				pubtest = source.split('<arxiv:doi')[1].split('</arxiv:doi>')[0].split('atom">')[1]
				pubtest = "DOI:"+pubtest
			except Exception as e2:
				#print (abst_id, e2)
				pubtest = "unpublished"

		testTmp = title+" "+abstract
		s=set(stopwords.words('english'))
		testTmp_2 = filter(lambda w: not w in s,testTmp.split())
		testTmp_3 = ' '.join(map(str,testTmp_2) )

		tmp = [abst_id, auth_str, pubtest, title, testTmp_3, prim_cat]
		df_tmp = pd.DataFrame([tmp], columns=dfKeys)
		dfmonth = dfmonth.append(df_tmp,ignore_index=True)
	print (len(dfmonth), end=",")
	sys.stdout.flush()
	return dfmonth


if __name__ == "__main__":
	main(sys.argv[1:])



