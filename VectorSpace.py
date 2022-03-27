from distutils.util import copydir_run_2to3
from enum import unique
from pprint import pprint
from pydoc import doc
from Parser import Parser
import util
from operator import itemgetter
import numpy
import string
import nltk
import math
from nltk.stem import WordNetLemmatizer

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    tfidfdocumentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]

    documentNames = []

    termIdfIndex = {}


    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)
    def setdocumentnames(self,documents):
        self.documentNames = documents
    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]
        self.termIdfIndex = self.getTermIdfIndex(documents)
        self.tfidfdocumentVectors = self.gettfidfdocumentVectors(self.documentVectors)

    def gettfidfdocumentVectors(self,documentVectors):
        tfidfdocumentVectors = []
        for documentVector in documentVectors:
            tfidfdocumentVectors.append([a*b for a,b in zip(documentVector,self.termIdfIndex.values())])
        #print(tfidfdocumentVectors)
        return tfidfdocumentVectors
    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """

        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)
        vocabularyString = vocabularyString.translate(str.maketrans('', '', string.punctuation))

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)
    
    def getTermIdfIndex(self, documentList):
        vocab_documentList = []
        for document in documentList:
            document = document.translate(str.maketrans('', '', string.punctuation))
            vocabularyList = self.parser.tokenise(document)
            vocabularyList = self.parser.removeStopWords(vocabularyList)
            uniqueVocabularyList = util.removeDuplicates(vocabularyList)
            vocab_documentList.append(uniqueVocabularyList)


        termFrequencyIndex = {}
        for term in self.vectorKeywordIndex:
            termFrequencyIndex[term] = 0
            for vocab_document in vocab_documentList:
                if term in vocab_document:
                    termFrequencyIndex[term]+=1
            
            termFrequencyIndex[term] = math.log(len(documentList) / termFrequencyIndex[term])
        
        #print(termFrequencyIndex)
        return termFrequencyIndex
            


    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            try:
                vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
            except:
                print("term out of scope!: " + word)
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings


    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)
        #print(queryVector)
        #print(self.documentVectors)
        #input("breakpoint")
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]

        #ratings.sort(reverse=True)
        return ratings
    
    def tqCosine(self,searchList):
        result = self.search(searchList)
        tuple_list = list(zip(self.documentNames, result))
        tuple_list.sort(key = lambda x :x[1], reverse=True)
        return tuple_list[0:10]
    
    def tqEuc(self,searchList):
        queryVector = self.buildQueryVector(searchList)
        ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        tuple_list = list(zip(self.documentNames, ratings))
        return sorted(tuple_list,key=itemgetter(1))[:10]
    
    def tfidfCosine(self,searchList):
        queryVector = self.buildQueryVector(searchList)
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.tfidfdocumentVectors]
        #print(ratings)
        
        tuple_list = list(zip(self.documentNames, ratings))
        tuple_list.sort(key = lambda x :x[1], reverse=True)
        return tuple_list[0:10]

    def tfidfEuc(self,searchList):
        queryVector = self.buildQueryVector(searchList)
        #self.documentVectors = tfidf(self.documentVectors)
        ratings = [util.euclidean(queryVector, documentVector) for documentVector in self.tfidfdocumentVectors]
        tuple_list = list(zip(self.documentNames, ratings))
        return sorted(tuple_list,key=itemgetter(1))[:10]

    def feedback(self,searchList):
        queryVector = self.buildQueryVector(searchList)
        #self.documentVectors = tfidf(self.documentVectors)
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.tfidfdocumentVectors]
        tuple_list = list(zip(self.documentNames, ratings, self.documentVectors))
        feedbacktuple = sorted(tuple_list,key=itemgetter(1),reverse=True)[0]
        filename = feedbacktuple[0] + ".txt"
        feedback = ""
        with open("EnglishNews/" + filename, 'r') as fd:
            feedback = fd.read()

        text = nltk.word_tokenize(feedback)
        pos_tagged = nltk.pos_tag(text)


        newterms = ""
        for term,type in pos_tagged:
            tags = set(['VB','VBD','VBG','VBN','VBP','VBZ','NN','NNS','NNP','NNPS'])
            if type in tags and len(term) > 1:
                newterms += term
                newterms += " "

        newterms =newterms.translate(str.maketrans('', '', string.punctuation))


        feedbackVector = self.buildQueryVector([newterms])
        
        for i in range(0,len(feedbackVector)):
            queryVector[i] = feedbackVector[i] * 0.5 + queryVector[i]
        
        #self.documentVectors = tfidf(self.documentVectors)
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.tfidfdocumentVectors]
        tuple_list = list(zip(self.documentNames, ratings))
        return sorted(tuple_list,key=itemgetter(1),reverse=True)[:10]

def tfidf(documentVectors):
    frequencyIndex = []
    for index in range(0,len(documentVectors[0])):
        count = 0
        for i in range(0,len(documentVectors)):
            if documentVectors[i][index] > 0:
                count+=1
        frequencyIndex.append(count)
    
    frequencyIndex = [ math.log(len(documentVectors)/i)  for i in frequencyIndex]

    #print(frequencyIndex)
    for index in range(0,len(documentVectors)):
        documentVectors[index] = [a*b for a,b in zip(documentVectors[index],frequencyIndex)]
    return documentVectors


def prettyprint(tuple_list,label):
    print('-----------------------------')
    print(label)
    print("")
    print("NewsID              Score")
    print("------              -----")
    for(id,score) in tuple_list:
        print(id,"        ",score)

def check(num,tuple_list):
    my_list = []
    count = 0
    if num == 1:
        my_list = ["News123256","News119356","News111959","News115859","News120265","News119746","News101763","News108578","News107163","News122750"]
    if num == 2:
        my_list = ["News107883","News108482","News109808","News110033","News110141","News110871","News108024","News108653","News108964","News110211"]
    if num == 3:
        my_list = ["News108813","News104913","News116613","News103134","News116634","News103728","News110804","News121995","News118108","News103767"]
    if num == 4:
        my_list = ["News107883","News110329","News110871","News108482","News105142","News110514","News110033","News110804","News110141","News111579"]
    if num == 5:
        my_list = ["News200049","News200892","News200847","News200908","News200161","News200137","News200056","News200565","News200071","News200898"]
    for (newsid,score) in tuple_list:
        if newsid in my_list:
            count+=1
    return count
import os
import re
from tqdm import tqdm
from nltk import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
nltk.download('wordnet')
lancaster=LancasterStemmer()
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(lancaster.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

import argparse
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description= 'Input a string.')
    parser.add_argument('--query',dest= 'sentence', nargs= '+')
    parser.add_argument('--option',dest= 'option', nargs= '+')
    parser.add_argument('--source',dest= 'source', nargs= '+')
    args = parser.parse_args()
    source = args.source[0]
    print("News Source Folder: ",source)
    option = int((args.option)[0])
    sentence = [' '.join(args.sentence)]
    print("Query: ",sentence)
    #test data
    '''
    documents = ["The cat in the hat disabled",
                 "A cat is a fine pet ponies.",
                 "Dogs and cats make good pets.",
                 "I haven't got a hat.",
                 "cat cat cat"]
    '''
    documents = []
    file_name_list = []
    all_files = os.listdir(source) 
    #print(all_files)
    for file in tqdm(all_files): #[:10] for debug purposes
        #print("EnglishNews/" + file)
        with open(source + file, 'r') as fd:
            content = fd.read()
            readinfo = content
            content = content.replace('\n',' ')
            content = content.replace('“','')
            content = content.replace('”','')
            content = content.translate(str.maketrans(' ', ' ', string.punctuation)).strip() #remove punctuation here
            documents.append(content)
            #print(content)
        file_name_list.append(file[:-4])
    
    vectorSpace = VectorSpace(documents)
    

    vectorSpace.setdocumentnames(file_name_list)

    #print(vectorSpace.vectorKeywordIndex.keys())

    #print(vectorSpace.documentVectors)

    #print(vectorSpace.related(1))
    if option == 1:
        result = vectorSpace.tqCosine(sentence)
        prettyprint(result,"(1) Term Frequency (TF) Weighting + Cosine Similarity")
        #print("correct: ", check(1,result), "/10")
    elif option == 2:
        result = vectorSpace.tqEuc(sentence)
        prettyprint(result,"(2) Term Frequency (TF) Weighting + Euclidean Distance")
        #print("correct: ", check(2,result), "/10")
    elif option == 3:
        result = vectorSpace.tfidfCosine(sentence)
        prettyprint(result,"(3) TF-IDF Weighting + Cosine Similarity")
        #print("correct: ", check(3,result), "/10")
    elif option == 4:
        result = vectorSpace.tfidfEuc(sentence)
        prettyprint(result,"(4) TF-IDF Weighting + Euclidiean Distance")
        #print("correct: ", check(4,result), "/10")
    elif option == 5:
        result = vectorSpace.feedback(sentence)
        prettyprint(result,"(5) Term Frequency (TF-IDF) Weighting + Cosine Similarity with Feedback")
    else:
        print('please input a number from 1~5')



###################################################
