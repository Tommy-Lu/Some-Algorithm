from numpy import *
#import operator

"""
函数名：loadDataSet(fileName)
输	入：文本文件等。
功	能：将文本文件中每行数据读出，进行浮点转换后，将每一行作为一个列表，添加到一个矩阵列表中。
返回值：元素为列别的列表。
"""
def loadDataSet(fileName):
	print(fileName)
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		#print(line)
		curLine = line.strip().split('\t')
		#print(curLine)
		fltLine = list(map(float, curLine))   #map映射，将curLine全部按照float转化，得到一个列表。python3中需要添加list才能得到list的表。
		dataMat.append(fltLine)		#将列表添加到列表中。
		#dataMat.append(curLine)		#将列表添加到列表中。
		#print(fltLine, dataMat)
	return dataMat

"""
函数名：distEclud(vecA, vecB)
输	入：两个向量
功	能：欧氏距离公式，计算两个向量之间的距离
		a = (a1, a2) b =(b1, b2) d(a,b) = sqrt( (a1 - a2)*(a1 - a2) + (b1 - b2)*(b1 - b2) )
返回值：返回两个向量之间的距离
"""
def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))

"""
函数名：randCent(dataSet, k)
输	入：数据集列表，随机取点的个数k
功	能：在给定的数据集中构建一个包含k个随机质心的集合，随机质心必须要在数据集的边界之内。
		可以通过数据集中的每一维的最小和最大值来完成。
返回值：随机选取的k个质心的数据集
"""
def randCent(dataSet, k):
	n = shape(dataSet)[1]   #通过shape函数，求出dataset的矩阵的列数大小
	centroids = mat(zeros((k, n)))	#构建一个k行n列的0矩阵。 zeros构建数组，通过numpy的mat函数将其转化为矩阵模式。即存储k个质心的矩阵
	for j in range(n):				#遍历矩阵的所有列，
		minJ = min(dataSet[:, j])	#取出列中最小值
		rangeJ = float(max(dataSet[:, j]) - minJ)	#计算列的值的宽度，最大值减最小值
		centroids[:, j] = minJ + rangeJ * random.rand(k, 1)  #产生一个k行1列的随机数，乘上列的宽度，加上最小值，得到一组列范围内的随机取值，作为质心的一个列值
	return centroids   #循环n列后，将得到一个k行n列的质心的新的矩阵。
	
"""
相关说明
>>> numpy.shape(aa)   #shape函数计算行列
(3, 3)
>>> numpy.random.rand(k, n) #产生一个k行n列的1以内的随机数
>>> numpy.random.rand(5,2)
array([[0.35459977, 0.25003408],
       [0.98617748, 0.08624928],
       [0.53547044, 0.31481714],
       [0.21593504, 0.30125012],
       [0.45701022, 0.21920715]])
>>> a    #转化为矩阵
[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
>>> numpy.mat(a)
matrix([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

>>> aa		#矩阵中计算列中所有数值
matrix([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])
>>> aa[:,1]		#获取矩阵中第一列的所有值
matrix([[2],
        [5],
        [8]])
"""
	
"""
函数名：kMeans(dataSet, k, distMeas=distEclud, createCent=randCent)
输	入：待处理数据集dataSet，其实设置的质心个数k，计算距离函数distMeas和起始创建之心函数createCent有使用默认输入即可。
功	能：将输入数据集dataSet按照预设k个类进行聚类操作。当每个元素所属的簇分类不在变化时，说明聚类操作完成，结束函数。
返回值：centroids分类质心的位置向量矩阵，clusterAssment，每个元素所属质心簇索引及该元素到该簇质心之间的距离的矩阵。
"""
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	m = shape(dataSet)[0]		#获取数据集行数
	clusterAssment = mat(zeros((m, 2)))	#创建m行2列的矩阵，用于存储dataSet中每个元素对应属于哪个簇，以及到该簇的距离。
	centroids = createCent(dataSet, k)	#计算dataSet的k个质心
	clusterChanged = True		#簇改变标识，用于循环执行检查
	while clusterChanged:
		clusterChanged = False
		for i in range(m):		#循环给出的数据集中所有的行
			minDist = inf 		#定义最小距离，numpy.inf为一个无穷大的初值
			minIndex = -1		#定义质心最小距离处索引位置
			for j in range(k):	#遍历每个质心 ，完成给出dataSet中每个元素到质心距离的计算，并找到最近距离的质心已经质心索引位置。
				distJI = distMeas(centroids[j, :], dataSet[i, :])	#计算质心中的每个元素与数据集中给出元素的距离
				if distJI < minDist:   	#找出所有质心中距离数据集元素最近的质心，更新最小距离，以及质心向量的索引位置
					minDist = distJI	#更新最小距离
					minIndex = j		#更新最小距离是的质心索引位置
			if clusterAssment[i, 0] != minIndex	:	#如果重新计算的所属簇和原来的不一样，则继续计算
				clusterChanged = True				#当这里的m行个元素所属的簇索引不在变化时，将会停止while循环，及停止计算新的质心
			clusterAssment[i, :] = minIndex, minDist**2		#更新clusterAssment矩阵中对应元素的所属簇索引以及到该簇距离值
		print (centroids)		#打印计算得到的质心向量
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]]	#取得某个簇索引下包含的所有点，存储在ptsInClust
			centroids[cent, :] = mean(ptsInClust, axis = 0)					#对该簇索引下的所有点重新求点质心（均值），并将新质心位置更新到centroids中
		
	return centroids, clusterAssment
"""
说明：
>>> nca  	#矩阵nca
matrix([[1, 3],
        [1, 3],
        [2, 4],
        [2, 3],
        [1, 4]])
>>> nca[:, 0]	#取矩阵的第0列元素
matrix([[1],
        [1],
        [2],
        [2],
        [1]])
>>> nca[:, 0].A	#将矩阵取得等于自身的数组
array([[1],
       [1],
       [2],
       [2],
       [1]])
>>> nca[:, 0].A == 1	#返回数组中值等于1的布尔型值
array([[ True],
       [ True],
       [False],
       [False],
       [ True]])
>>> numpy.nonzero(nca[:,0].A==1)	#获取数组中不为0的值得索引地址
(array([0, 1, 4], dtype=int32), array([0, 0, 0], dtype=int32))
>>> numpy.nonzero(nca[:,0].A==1)[0]
array([0, 1, 4], dtype=int32)

>>> x
matrix([[ 1, 12,  3],
        [23,  1,  2],
        [ 3,  4,  5]])
>>> numpy.mean(x, axis = 0)	#将矩阵x按照列的方式求均值
matrix([[9.        , 5.66666667, 3.33333333]])
>>>

"""	
	
"""
函数名：
输	入：
功	能：
返回值：
"""
def biKmeans(dataSet, k, distMeas=distEclud):
	m = shape(dataSet)[0]	#求出数据集中元素个数
	clusterAssment = mat(zeros((m, 2)))	#m行个2列的0矩阵，用于存放计算的数据集中每个元素属于簇编号，已经元素到该质心的距离
	#centroid0 = mean(dataSet, axis = 0).tolist()[0]	#将dataSet按照列取均值，得到一个
	centroid0 = mean(dataSet, axis = 0).tolist()	#将dataSet按照列取均值(得到质心)
	centList = [centroid0]	#质心坐标的列表
	for j in range(m):		#遍历数据集中每个元素
		clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2	#计算dataSet中每个元素到质心的距离，并将得到的值平方，存入clusterAssment对应于dataSet元素位置的矩阵中。
	while(len(centList) < k):	#当得到的质心数达到预设的k值时，停止循环。
		lowestSSE = inf			#设置最低SSE初始值为无穷大
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]	#计算dataSet中分类簇i下数据质心centList中簇索引的所有元素，存入ptsInCurrCluster
			centroidMat, splitCluster = kMeans(ptsInCurrCluster, 2, distMeas)		#将得到的属于同一质心的所有点进行K=2的聚类，并得到聚类后的质心坐标，和质心索引及各元素到质心的距离的矩阵
			sseSplit = sum(splitCluster[:, 1])	#计算得到的质心索引-元素到质心距离矩阵中的距离的平方的和。
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])	#计算为分类簇i下的所有元素到质心的距离和。
			print("sseSplit, and notSplit:", sseSplit, sseNotSplit)
			if (sseSplit + sseNotSplit) < lowestSSE:	#跟新最小SSE值
				bestCentToSplit = i			#簇分类最优的簇标识
				bestNewCents = centroidMat	#簇分类最优的质心
				bestClustAss = splitCluster.copy()
				lowestSSE = sseSplit + sseNotSplit	#跟新最小SSE值
		
		bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
		bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
		
		print("the bestCentToSplit is :", bestCentToSplit)
		print("the len of bestClustAss is :", len(bestClustAss))
		
		centList[bestCentToSplit] = bestNewCents[0, :]
		centList.append(bestNewCents[1, :])
		clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
	
	return 	centList, clusterAssment

"""
函数名：
输	入：
功	能：
返回值：
"""	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	