"""
文件：DataConvert.py
功能：将输入的文件夹中的文件，按照文件名中标注的类型进行分类，并将文件中的内容(中文字符)进行数字化转换，保存文件。
使用：python DataConvert.py  inpath  OutFileName   #将inpath中的文件按照80%的比例分为OutFileName.train 和OutFileName.test两个文件
输入文件夹中文件格式如下：12business.seg.cln.txt 148auto.seg.cln.txt 358sports.seg.cln.txt 
"""
import sys
import os
import random

WordList = []
WordIDDic = {}
TrainingPercent = 0.8   #设置训练和测试文档的比例

inpath = sys.argv[1]    #从sys.arv[1]获取inpath的路径
OutFileName = sys.argv[2]  #从sys.arv[2]获取output的路径文件名
trainOutFile = file(OutFileName+".train", "w")
testOutFile = file(OutFileName+".test", "w")

def ConvertData():
    i = 0
    tag = 0
	#os.listdir列出文件路径下所有的文件名
    for filename in os.listdir(inpath):
        if filename.find("business") != -1:   #字符串查找，查找到指定字符串时，返回非-1值，否则返回-1
            tag = 1    #从文件名来判断并标记文件的类型
        elif filename.find("auto") != -1:
            tag = 2
        elif filename.find("sport") != -1:
            tag = 3
        i += 1
        rd = random.random()   #随机函数，用于产生0.8比例的数据
        outfile = testOutFile  #设置输出文件
        if rd < TrainingPercent:  #当random值小于0.8时，输出到训练文档中。
            outfile = trainOutFile

        if i % 100 == 0:
            print i,"files processed!\r",

        infile = file(inpath+'/'+filename, 'r')  #打开遍历到的文件
        outfile.write(str(tag)+" ")      #将得到的文件类型写入到输出文档中。
        content = infile.read().strip()
        content = content.decode("utf-8", 'ignore')
        words = content.replace('\n', ' ').split(' ')   #将文档中'\n'替换为' '，并且按照' '进行切分
        for word in words:
            if len(word.strip()) < 1:  #排除字符长度异常的值
                continue
            if word not in WordIDDic:  #将新词添加到WordIDDic
                WordList.append(word)  #添加新词到WordList
                WordIDDic[word] = len(WordList)  #这里为新词分配一个唯一的数字标识len(WordList)，因为添加新词后，len(WordList)就会增大
            outfile.write(str(WordIDDic[word])+" ")  #将字符转换成数字标识，写入输出文件中。
        outfile.write("#"+filename+"\n")    #将文件名写入输出文件中。
        infile.close()

    print i, "files loaded!"
    print len(WordList), "unique words found!"

ConvertData()
trainOutFile.close()
testOutFile.close()
