from math import log
import operator

#计算给定数据集的熵
#使用公式：H =-∑_(i=1)^n〖p(x_i )log2(px_i)〗
#返回熵值，熵值越大，则混合的数据种类也就越多，相反熵值越小，则数据种类就越少
#最优的结果是，当前数据集中只有一个特征，即熵值为0
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)   #计算整个数据集中类别总数
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]   # 从数据集中获取label
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1   #计算每种label的数量，用于计算此label出现的概率，即p(xi) 
	shannonEnt = 0.0   #设定香农熵的初值
	for key in labelCounts.keys():
		prob = float(labelCounts[key])/ numEntries   #计算p(xi)值
		shannonEnt -= prob * log(prob, 2)  #按照求熵的公式求熵值
	
	return shannonEnt  #返回熵值
	
def createDataSet():
	dataSet = [ [1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']
	]
	
	labels = ['no surfacing', 'flippers']

	return dataSet, labels

#dataSet：待划分的数据集
#axis：划分数据集的特征
#value：特征的返回值
#将数据集dataSet按照特征axis，value来划分，产生新的数据集，并返回
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis + 1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

#dataSet输入待划分的数据集
#功能：将输入的数据集进行划分，并找到最好的划分方式并返回最优划分时的特征值。
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1 #获取特征种类数量，注意这里-1是去掉最后的label标签
	baseEntropy = calcShannonEnt(dataSet)  #计算最原始的数据集熵，此时熵值最大
	bestInfoGain = 0.0   #设定最优的增益值 
	bestFeature = -1	 #设置默认的最优特征值
	
	for i in range(numFeatures):   #特征种类数循环
		featList = [example[i] for example in dataSet]   #此处将某个特征下所有的值取出来存放在列表中。
		uniqueVals = set(featList)  #将列表中的值进行去重
		newEntropy = 0.0  #定义新的熵值
		for value in uniqueVals:  #循环某个特征下的值，划分数据集，并计算新的熵值
			subDataSet = splitDataSet(dataSet, i, value)  #划分数据集
			prob = len(subDataSet) / float(len(dataSet))  #计算新熵： 新数据集中特征种类 / 原始数据集中特征种类
			newEntropy += prob * calcShannonEnt(subDataSet) #计算新熵，将所有新熵值求和，注意此处有prob可以理解为此新数据集的前置概率
		infoGain = baseEntropy - newEntropy  #熵值越小，特征种类越少，此时得到的infoGain值就越大
		if (infoGain > bestInfoGain):  #每个特征计算出来的infoGain都和上一次的进行比较，当此值比上次的大时，即此次划分的数据集特征种类最少，更新bestInfoGain
			bestInfoGain = infoGain    #更新bestInfoGain
			bestFeature = i			   #获取此时的特征索引值	
	return bestFeature    #返回最优划分时的特征索引值

#计算输入分类list中，分类最多的类，并返回
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	#利用sorted函数对classCount字典进行排序，并返回分类最终的类。
	#注意，使用sorted函数后等到一个列表，在中列表中取出排序在最前面的字典，并取字典的key值()返回。
	sortedClassCount = sorted(classCount.iteritems(), key= operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]	

#输入待分类数据集dataSet
#输入待分类数据集的标签分类信息
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]  #获取输入数据集的所有类标签
	
	if classList.count(classList[0]) == len(classList): #判断数据集中类标签完全相同，则直接返回该类标签。
		return classList[0]
		
	if len(dataSet[0]) == 1:   #如果数据集中只剩下唯一类别的分组，则调用出现次数最多的类标签作为其返回类标签。
		return majorityCnt(classList)
	
	bestFeat = chooseBestFeatureToSplit(dataSet)   #对数据集中划分计算并得到最好的特征变量
	bestFeatLabel = labels[bestFeat]    #从labels中选取最好特征变量的值
	myTree = {bestFeatLabel:{}}   #构建用于输出的树形字典
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]  # 在数据集中选取最好特征下的所有值，用于再次计算
	uniqueVals = set(featValues)    #去重处理
	for value in uniqueVals:   #对所有值进行循环处理，用于再次创建决策树
		subLabels = labels[:]  #labels标签的复制，重新存储，防止label在计算的时候被改变。 
		#递归调用createTree函数自身，直到满足上边两个if判断的条件，递归才退出。
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	
	return myTree  #返回得到的决策树
	
	
	