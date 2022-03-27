#http://tartarus.org/~martin/PorterStemmer/python.txt
from PorterStemmer import PorterStemmer
import jieba, re
class Parser:

	#A processor for removing the commoner morphological and inflexional endings from words in English
	stemmer=None

	stopwords=[]

	def __init__(self,):
		self.stemmer = PorterStemmer()

		#English stopwords from ftp://ftp.cs.cornell.edu/pub/smart/english.stop
		self.stopwords = open('english.stop', 'r').read().split() + open('chinese_proj1.stop', 'r').read().split() 
		#print(self.stopwords)

	def clean(self, string):
		""" remove any nasty grammar tokens from string """
		string = string.replace(".","")
		string = string.replace("\s+"," ")
		string = string.lower()
		return string
	

	def removeStopWords(self,list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in self.stopwords ]


	def tokenise(self, string):
		""" break string up into tokens and stem words """
		string = self.clean(string)
		words = string.split(" ")
		wordlist = [self.stemmer.stem(word,0,len(word)-1) for word in words]
		newwordlist = []
		for word in wordlist:
			newwordlist += self.tokeniseChinese(word)
		return newwordlist

	def tokeniseChinese(self, string):
		if re.search(u'[\u4e00-\u9fff]', string):
			#print('found Chinese: ', string)
			wordlist = list(jieba.cut(string, cut_all=False, HMM=True))
			wordlist = filter(lambda term: term != ' ' and len(term) > 1, wordlist)
			return wordlist
		else:
			return [string]

