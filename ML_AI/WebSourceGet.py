import urllib.request
import string
import sys
import _pickle as cPickle
import os

# - This script is used to download all the data from arXiv.org for a given year. 
def main(argv):

	# - input year label - 
	try:
		year = argv[0]
	except Exception as e:
		print (e)

	print("Getting Data from ArXiv API Year = ", year)
	os.system("mkdir -p WebSources")

	# defining month array for the API 
	monthsval = ['01','02','03','04','05','06','07','08','09','10','11','12']

	for month in monthsval:

		# Building the API url    CS: Machine learning
		url_start = "http://export.arxiv.org/api/query?search_query=cat:cs.LG*+AND+submittedDate:["
		url_end = "9999]&start=0&max_results=10000"
		url_mid = str(year) + str(month) + "010000+TO+" + str(year) + str(month) + "31"
		print ("   month/year = "+str(month)+"/"+str(year), end=" ")
		#print ("   ", url_start+url_mid+url_end)
		print ("    CS: Machine Learning ", end=" ")

		# Read the source html input 
		source = urllib.request.urlopen(url_start+url_mid+url_end).read()


		# Building the API url --- CS: Artificial intelligence
		url_start = "http://export.arxiv.org/api/query?search_query=cat:cs.ai*+AND+submittedDate:["
		url_end = "9999]&start=0&max_results=10000"
		url_mid = str(year) + str(month) + "010000+TO+" + str(year) + str(month) + "31"
		#print ("   month/year = "+str(month)+"/"+str(year), end=" ")
		#print ("   ", url_start+url_mid+url_end)
		print ("    CS: Artificial intelligence ", end=" ")

		# Read the source html input 
		source += urllib.request.urlopen(url_start+url_mid+url_end).read()



		# Building the API url ---- Stat: Machine Learning
		url_start = "http://export.arxiv.org/api/query?search_query=cat:stat.ml*+AND+submittedDate:["
		url_end = "9999]&start=0&max_results=10000"
		url_mid = str(year) + str(month) + "010000+TO+" + str(year) + str(month) + "31"
		#print ("   month/year = "+str(month)+"/"+str(year), end=" ")
		#print ("   ", url_start+url_mid+url_end)
		print ("    Stat: Machine Learning ", end=" ")

		# Read the source html input 
		source += urllib.request.urlopen(url_start+url_mid+url_end).read()



		# dump the source html raw data into outfile
		outfile = "WebSources/Source_"+str(year)+"_"+str(month)+".txt"		
		cPickle.dump(source, open(outfile, 'wb'))
		#events = source.split('<entry>')
		print ("   Events length: ", len(source))

if __name__ == "__main__":
    main(sys.argv[1:])



