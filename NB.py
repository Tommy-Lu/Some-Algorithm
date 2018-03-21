"""
文件：NB.py
功能：将训练集文档输入，通过朴素贝叶斯公式，计算得到一个model；将测试集文档输出到model，得到一个测试结果，并通过这个结果来计算这个模型用来分类的精确度、针对某个类的精准度以及针对某个类的召回率。
使用： python NB.py 1 TrainingDataFile ModelFile
       python NB.py 0 TestDataFile ModelFile OutFile
"""
#Usage:
#Training: NB.py 1 TrainingDataFile ModelFile
#Testing: NB.py 0 TestDataFile ModelFile OutFile

import sys
import os
import math


DefaultFreq = 0.1
TrainingDataFile = "nb_data.train"
ModelFile = "nb_data.model"
TestDataFile = "nb_data.test"
TestOutFile = "nb_data.out"
ClassFeaDic = {}   #用于存放类标签及此类标签下存放的所有词数量
ClassFreq = {}     #用于存放类标签及类标签出现的次数
WordDic = {}	   #用于存放所有文档中出现的词，不重复的词集
ClassFeaProb = {}		#用于存放类标签下，所有词出现的概率
ClassDefaultProb = {}	#存放未出现类标签的默认概率
ClassProb = {}			#存放类标签的概率

def Dedup(items):
    tempDic = {}
    for item in items:
        if item not in tempDic:
            tempDic[item] = True
    return tempDic.keys()

"""
函数名：LoadData()
功	能：计算各类别标签的数量，存放在ClassFreq字典中。
		计算所有文档下所有词的无重复组合，存放在WordDic字典中。
		计算各类别下，各个词出现的次数，并记录保存在ClassFeaDic中。
"""
def LoadData():
    i =0
    infile = file(TrainingDataFile, 'r')
    sline = infile.readline().strip()   #载入训练集数据，其中每行为一个文档的内容，每行以' '分割，第一个字符代表类别标签。
    while len(sline) > 0:
        pos = sline.find("#")		#定位文件名处位置
        if pos > 0:
            sline = sline[:pos].strip()		#去掉多余加上的文件名
        words = sline.split(' ')			#获取到需要的内容部分
        if len(words) < 1:		#去除异常部分
            print "Format error!"
            break
        classid = int(words[0])			#获取类别标签
        if classid not in ClassFeaDic:	#初始化类别标签下的初始值
            ClassFeaDic[classid] = {}	#字典嵌套字典，字典的值为一个字典
            ClassFeaProb[classid] = {}
            ClassFreq[classid]  = 0 	#
        ClassFreq[classid] += 1			#累加类别的数量
        words = words[1:]				#内容部分重新赋值
        #remove duplicate words, binary distribution
        #words = Dedup(words)
        for word in words:
            if len(word) < 1:	#去掉异常长度的word
                continue
            wid = int(word)		#类型转换
            if wid not in WordDic:	
                WordDic[wid] = 1	#新出现的词，添加到WordDic字典，无重复key字典
            if wid not in ClassFeaDic[classid]:  #将新词添加到ClassFeaDic[classid]中，如果词出现过，则累加计数。
                ClassFeaDic[classid][wid] = 1
            else:
                ClassFeaDic[classid][wid] += 1
        i += 1
        sline = infile.readline().strip()
    infile.close()
    print i, "instances loaded!"
    print len(ClassFreq), "classes!", len(WordDic), "words!"
	
"""
函数名：ComputeModel()
功	能：计算得到各个类别标签的概率，并存放在ClassProb字典中。
		计算得到各个类别标签下，各个词出现的概率，并存储在ClassFeaProb字典中。
		计算默认的类别频率。不知道何用？？存放在ClassDefaultProb字典中
"""
def ComputeModel():
    sum = 0.0
    for freq in ClassFreq.values():
        sum += freq		#计算总的类别标签数
    for classid in ClassFreq.keys():
        ClassProb[classid] = (float)(ClassFreq[classid])/(float)(sum)		#计算类别标签的概率，并存放在ClassProb字典中。
    for classid in ClassFeaDic.keys():
        #Multinomial Distribution
        sum = 0.0
        for wid in ClassFeaDic[classid].keys():
            sum += ClassFeaDic[classid][wid]			#计算某一个类别标签下所有词的总数
        newsum = (float)(sum+len(WordDic)*DefaultFreq)	#sum的排0处理。
        #Binary Distribution
        #newsum = (float)(ClassFreq[classid]+2*DefaultFreq)
        for wid in ClassFeaDic[classid].keys():
            ClassFeaProb[classid][wid] = (float)(ClassFeaDic[classid][wid]+DefaultFreq)/newsum	#计算类别标签下，某个词出现的概率，并存储在ClassFeaProb字典中。
        ClassDefaultProb[classid] = (float)(DefaultFreq) / newsum			#默认的类别频率。不知道何用？？
    return

"""
函数名：SaveModel()
c将得到的类别标签概率(p(ci))，类标签下各个词出现的概率(p(xj|ci))输出到文档ModelFile中。
"""
def SaveModel():
    outfile = file(ModelFile, 'w')
    for classid in ClassFreq.keys():	#存放类别标签 ，类别概率，默认概率三个值
        outfile.write(str(classid))
        outfile.write(' ')
        outfile.write(str(ClassProb[classid]))
        outfile.write(' ')
        outfile.write(str(ClassDefaultProb[classid]))
        outfile.write(' ' )
    outfile.write('\n')					#转行
    for classid in ClassFeaDic.keys():			#类别标签下，出现所有词，及词对应出现的概率
        for wid in ClassFeaDic[classid].keys():
            outfile.write(str(wid)+' '+str(ClassFeaProb[classid][wid]))
            outfile.write(' ')
        outfile.write('\n')
    outfile.close()

"""
函数名：LoadModel()
功	能：将存储的model中的类别标签概率、类别标签下各词的概率读取出来。将各数据存储在相关的字典中。
"""
def LoadModel():
    global WordDic
    WordDic = {}
    global ClassFeaProb
    ClassFeaProb = {}
    global ClassDefaultProb
    ClassDefaultProb = {}
    global ClassProb
    ClassProb = {}
    infile = file(ModelFile, 'r')
    sline = infile.readline().strip()
    items = sline.split(' ')
    if len(items) < 6:
        print "Model format error!"
        return
    i = 0
    while i < len(items):
        classid = int(items[i])  #获取类别标签
        ClassFeaProb[classid] = {}
        i += 1
        if i >= len(items):
            print "Model format error!" 
            return
        ClassProb[classid] = float(items[i])	#获取类别标签概率
        i += 1
        if i >= len(items):
            print "Model format error!" 
            return
        ClassDefaultProb[classid] = float(items[i]) #获取默认类别标签概率
        i += 1
    for classid in ClassProb.keys():	#遍历类别标签列表，取出相关类别标签列表的行数据。
        sline = infile.readline().strip()	#读出一行
        items = sline.split(' ')
        i = 0
        while i < len(items):
            wid  = int(items[i])
            if wid not in WordDic:
                WordDic[wid] = 1
            i += 1
            if i >= len(items):
                print "Model format error!"
                return
            ClassFeaProb[classid][wid] = float(items[i])	#取出类别标签下词的概率值
            i += 1
    infile.close()
    print len(ClassProb), "classes!", len(WordDic), "words!"


"""
函数名：Predict()
功	能：计算给出的测试数据集，计算每个数据属于的类别标签，并将预测的类别标签列表和实际的类别标签列表返回
"""
def Predict():
    global WordDic
    global ClassFeaProb
    global ClassDefaultProb
    global ClassProb

    TrueLabelList = []
    PredLabelList = []
    i =0
    infile = file(TestDataFile, 'r')	#测试数据集读入
    outfile = file(TestOutFile, 'w')	#测试输出测试集实际类型值和预测出来的类型值
    sline = infile.readline().strip()	#单行读入
    scoreDic = {}
    iline = 0
    while len(sline) > 0:
        iline += 1
        if iline % 10 == 0:
            print iline," lines finished!\r",
        pos = sline.find("#")	#读取文件名的位置。
        if pos > 0:
            sline = sline[:pos].strip()	#测试文件去掉文件名
        words = sline.split(' ')
        if len(words) < 1:
            print "Format error!"
            break
        classid = int(words[0])	#获取类别标签
        TrueLabelList.append(classid)	#添加实际标签到TrueLabelList列表
        words = words[1:]	#去掉类别标签后的内容
        #remove duplicate words, binary distribution
        #words = Dedup(words)
        for classid in ClassProb.keys():
            scoreDic[classid] = math.log(ClassProb[classid])	#类别标签概率对数化
        for word in words:
            if len(word) < 1:
                continue
            wid = int(word)
            if wid not in WordDic:
                #print "OOV word:",wid
                continue
            for classid in ClassProb.keys():
                if wid not in ClassFeaProb[classid]:
                    scoreDic[classid] += math.log(ClassDefaultProb[classid])	#未出现在词列表中的词，直接使用默认概率来做对数。
                else:
                    scoreDic[classid] += math.log(ClassFeaProb[classid][wid])	#求类别标签下各词概率的对数
					#注意这里：使用scoreDic[classid] += 是因为P(X_j |c_i)的连乘，当转换为对数后，使用加法。这样可以得到给出的测试数据集在那种类别标签下的概率最大。
        #binary distribution
        #wid = 1
        #while wid < len(WordDic)+1:
        #   if str(wid) in words:
        #       wid += 1
        #       continue
        #   for classid in ClassProb.keys():
        #       if wid not in ClassFeaProb[classid]:
        #           scoreDic[classid] += math.log(1-ClassDefaultProb[classid])
        #       else:
        #           scoreDic[classid] += math.log(1-ClassFeaProb[classid][wid])
        #   wid += 1
        i += 1
        maxProb = max(scoreDic.values())	#取出最大概率类别标签值
        for classid in scoreDic.keys():
            if scoreDic[classid] == maxProb:	#得到预测的类别标签。
                PredLabelList.append(classid)	#添加到预测标签列表。
        sline = infile.readline().strip()		#取下一行测试数据。
    infile.close()
    outfile.close()
    print len(PredLabelList),len(TrueLabelList)
    return TrueLabelList,PredLabelList

"""
函数名：Evaluate(TrueList, PredList)
功	能：计算预测正确标签和实际标签总数，得到预测准确度。
"""
def Evaluate(TrueList, PredList):
    accuracy = 0
    i = 0
    while i < len(TrueList):
        if TrueList[i] == PredList[i]:
            accuracy += 1
        i += 1
    accuracy = (float)(accuracy)/(float)(len(TrueList))
    print "Accuracy:",accuracy

"""
函数名：CalPreRec(TrueList,PredList,classid)
功	能：通过给定的预测标签和实际标签列表，和给出类别标签，计算某个类别的预测精确度和召回值。
"""
def CalPreRec(TrueList,PredList,classid):
    correctNum = 0
    allNum = 0
    predNum = 0
    i = 0
    while i < len(TrueList):
        if TrueList[i] == classid:
            allNum += 1
            if PredList[i] == TrueList[i]:
                correctNum += 1
        if PredList[i] == classid:
            predNum += 1
        i += 1
    return (float)(correctNum)/(float)(predNum),(float)(correctNum)/(float)(allNum)

#main framework
if len(sys.argv) < 4:
    print "Usage incorrect!"
elif sys.argv[1] == '1':
    print "start training:"
    TrainingDataFile = sys.argv[2]
    ModelFile = sys.argv[3]
    LoadData()
    ComputeModel()
    SaveModel()
elif sys.argv[1] == '0':
    print "start testing:"
    TestDataFile = sys.argv[2]
    ModelFile = sys.argv[3]
    TestOutFile = sys.argv[4]
    LoadModel()
    TList,PList = Predict()
    i = 0
    outfile = file(TestOutFile, 'w')
    while i < len(TList):
        outfile.write(str(TList[i]))
        outfile.write(' ')
        outfile.write(str(PList[i]))
        outfile.write('\n')
        i += 1
    outfile.close()
    Evaluate(TList,PList)
    for classid in ClassProb.keys():
        pre,rec = CalPreRec(TList, PList,classid)
        print "Precision and recall for Class",classid,":",pre,rec
else:
    print "Usage incorrect!"
