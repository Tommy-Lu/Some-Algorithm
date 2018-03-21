from numpy import *
from math import log
import operator

"""
loadDataSet：构建测试用的输入数据和该数据的被别标签
postingList:测试输入数据列表，相当于输入文档
classVec:该输入所有文档的类别标签，这些标签为已知，对应于每个输入文档。
"""
def loadDataSet():
	postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
				   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
	]
	
	classVec = [0, 1, 0, 1, 0, 1]    #1代表侮辱性文字，0代表正常言论
	
	return postingList, classVec
	

"""
createVocabList : 将输入的dataSet数据集，使用集合方式取其中所有单词，并以列表的方式返回。
其实这里就是一个将文档所有单词进行去重得到一个新列表的过程。
"""
"""
def createVocabList(dataSet):
		vocabSet = set([])
		for document in dataSet:
			vocabSet = vocabSet | set(document)
		return list(vocabSet)
"""
def createVocabList(dataSet):
	vocabSet = set()
	for document in dataSet:
		for word in document:
			vocabSet.add(word)
	return list(vocabSet)


"""
setOfWords2Vec : 将给出的inputSet数据列表，按照vocabList列表转换成向量形式
vocabList ： 所有文档的单词去重后的列表
inputSet ： 给出的需要词汇表进行转换成向量的输入数据集。
说明： 这里的例子只是用来判断输入的数据inputSet是否含有某个或者某些词，所以在没有考虑输入数据inputSet中是否包含了重复的词，即只考虑词是否出现，而不考虑词出现的多少。
"""		
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)  #创建和词汇表vocabList等长的列表，用于保存转换后的向量
	for word in inputSet:   #遍历输入的数据集
		if word in vocabList:   #如果数据数据集中的单词在词汇表中，则对应在词汇表中该词位置在returnVec中标记1
			returnVec[vocabList.index(word)] = 1
		else:
			print ("the word: %s is not in my vocabulary!" % word)
	
	return returnVec   #返回转换后的向量

"""
trainNB0：求出侮辱类型的概率，正常类型的概率
		   求出每个类型下各元素向量的概率。
trainMatrix ：输入的总的文档
trainCategory：对应于输入文档的类型标签
"""	
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs =  len(trainMatrix)    #计算总的输入文档数
	numWords = len(trainMatrix[0])		#输入的每篇文档中的元素个数，因为这里输入的是处理成向量形式的列表，所以每个文档的元素个数一样。
	pAbusive = sum(trainCategory)/float(numTrainDocs)  #求侮辱性类型的概率，因为侮辱性类型值为1，正常类型值为0，所以使用sum求和就得总侮辱性类型数，再除以总的文档数，就得侮辱性类型的概率
	p0Num = ones(numWords)	 #初始化p0Num，用于存放正常类型的文档的元素。zeros(5)=array([0., 0., 0., 0., 0.])	
	p1Num = ones(numWords)  #初始化p1Num，用于存放侮辱类型的文档的元素
	p0Denom = 2.0  #初始化p0Denom，用于存放正常文档所有元素和
	p1Denom = 2.0  #初始化p0Denom，用于存放正常文档所有元素和
	
	for i in range(numTrainDocs):	#遍历所有输入文档数量
		if trainCategory[i] == 1:	#找出对应类型为侮辱性的文档
			p1Num += trainMatrix[i]	#将得到的文档相加到p0Num，注意这里用到了数组向量加法：
			p1Denom += sum(trainMatrix[i]) #将得到的文档对元素求和，累加到p1Denom，用于计算
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])			
	
	#求每个元素向量在此类型下的概率。 直接使用数组除以浮点数，所以p1Denom/p0Denom要初始化为浮点数。
	#得到的p0Vect/p1Vect即使每个类型标签下各元素向量的概率。
	#print("p1Denom=%s, p0Denom=%s"% (p1Denom,p0Denom))
	p1Vect = p1Num / p1Denom
	#p1Vect = log(p1Vect)  #由于小数概率相乘，结果会越来越小，所以将概率进行对数转换
	p0Vect = p0Num / p0Denom
	#p0Vect = log(p0Vect)
		
	return p0Vect, p1Vect, pAbusive
			
"""		
向量加法
>>> z = zeros(5)
>>> z
array([0., 0., 0., 0., 0.])
>>> l = [1,2,3,4,5]
>>> l
[1, 2, 3, 4, 5]
>>> z += l
>>> z
array([1., 2., 3., 4., 5.])
>>>
向量除法  用数组除以一个浮点数即可。
array([1., 2., 3., 4., 5.])
>>>
>>> z / sum(z)
array([0.06666667, 0.13333333, 0.2       , 0.26666667, 0.33333333])
>>>
"""		
"""
Test:
 import bayes
 reload (bayes)
 listOfPosts,listClasses = bayes.loadDataSet()
 mvVocabList = bayes.createVocabList(listOfPosts)
 trainMat = []
 for postingDoc in listOfPosts:
	trainMat.append(bayes.setOfWords2Vec(mvVocabList, postingDoc))
 p0v, p1v , pAb = bayes.trainNB0(trainMat, listClasses)	
"""
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		