import sys
import math
#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	#print("vector1: ",vector1, " vector2: ",vector2)
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def euclidean(vector1, vector2):
	result = 0
	for i in range(0,len(vector1)):
		result += (vector1[i] - vector2[i])**2
	result = math.sqrt(result)
	return float(result)
	

	

