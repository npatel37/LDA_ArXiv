import urllib.request
import string
import sys
import _pickle as cPickle
import os


def main(argv):
	try:
		year = argv[0]
	except Exception as e:
		print (e)

	print(" Getting Data from ArXiv API Year = ", year)
	os.system("mkdir -p WebSources")
	monthsval = ['01','02','03','04','05','06','07','08','09','10','11','12']

	for month in monthsval:
		url_start = "http://export.arxiv.org/api/query?search_query=cat:cond-mat*+AND+submittedDate:["
		url_end = "9999]&start=0&max_results=10000"
		url_mid = str(year) + str(month) + "010000+TO+" + str(year) + str(month) + "31"
		outfile = "WebSources/Source_"+str(year)+"_"+str(month)+".txt"		
		print ("   ", url_start+url_mid+url_end)
		source = urllib.request.urlopen(url_start+url_mid+url_end).read()
		cPickle.dump(source, open(outfile, 'wb'))
		#events = source.split('<entry>')
		print ("   Events length: ", len(source))
		




if __name__ == "__main__":
    main(sys.argv[1:])



