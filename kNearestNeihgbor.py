"""
K近邻算法基础实现模拟
"""
import numpy as np
import matplotlib.pyplot as plt

"""
将输入数据test-data.txt转换成矩阵形式
返回x矩阵1000 x 3, y矩阵1000 x 1
"""
def file2list():
    xdata , ydata = [], []
    file = open('test-data.txt', 'r')
    for f in file.readlines():
        ss = f.strip().split('\t')
        xdata.append([float(ss[0]), float(ss[1]), float(ss[2])])
        ydata.append(ss[-1])
        
    file.close()
    return xdata, ydata

"""
将数据归一化
每个数据归一化的过程为： (x - min) / (max - min)
返回归一化后的新数据集
"""
def norDataset(data):
    maxData = np.max(data, axis = 0)
    minData = np.min(data, axis = 0)
    
    return (data - minData) / (maxData - minData)

"""
计算两个点之间的L1距离(Manhattan曼哈顿距离)
"""
def L1dis(data1, data2):
    return  np.sum(np.abs(data1 - data2))   

"""
计算两个点之间的L2距离(Euclidean欧几里得距离)
"""
def L2dis(data1, data2):
    return  np.sqrt(np.sum((data1 - data2)**2))

"""
计算测试样本点和训练集之间的k个最近邻
返回这k个最近邻训练样本的label中类别最多的标签
test: 需要测试预测的样本
trdata：训练样本集
trlabel：训练样本标签集
k：设置最近邻个数k
"""
def kNearestNeihgbor(test, trdata, trlabel, k):
    distance = {}    #用于保存k个最近邻值及其对应的label索引值
    yPre = []        #输出k个最近邻的lable
    for i in range(k):  #初始化k个最近邻值为无穷大
        distance[i] = float("inf")
        
    mumExample = len(trdata)   
    for i in range(mumExample):   #遍历悬链样本集中的每一个样本，计算和预测样本的距离
        newDis = L1dis(test, trdata[i])   #调用距离计算公式
        disMaxIndex = max(distance,key=distance.get)  #获取保存k个最近邻距离值中最大距离的key值，最开始key值是从0到k-1
        
        if newDis < distance[disMaxIndex]:   #新计算的距离值小于保存的k个近邻值中的任何一个，即可更新此距离和此样本的顺序编号到字典中保存
            del(distance[disMaxIndex])   #将最大值从字典中删除
            distance[i] = newDis         #更新值到字典中
    
    #计算返回的k个label中的各标签的比例，将其中占比最大的返回
    for i in distance.keys():
        yPre.append(trlabel[i])
   
    #将列表yPre中的元素进行处理，计算其中元素个数，并返回元素个数最多的元素
    result = {}
    for i in set(yPre):
        result[i] = yPre.count(i)
    return max(result, key=result.get)

"""
使用matplot来将所有数据点绘图出来
"""
def plotDataPoint(xdata, ydata):
    #label = set(ydata)
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    xdata_0 = xdata[ydata == "smallDoses" ]
    xdata_1 = xdata[ydata == "didntLike"]
    xdata_2 = xdata[ydata == "largeDoses"]
    
    plt.plot(xdata_0[:, 0], xdata_0[:, 2], "ro")
    plt.plot(xdata_1[:, 0], xdata_1[:, 2], "bs")
    plt.plot(xdata_2[:, 0], xdata_2[:, 2], "m^")
    
    #plt.xlabel('xdata')
    #plt.ylabel('ydata')
    plt.title("data points")
    plt.legend(["smallDoses", "didntLike", "largeDoses"] )
    plt.show()

"""
程序入口
"""
if __name__ == "__main__":
    xdata , ydata = file2list()   #将给到的数据文本转化为列表类型
    norXdata = norDataset(xdata)  #得到的样本数据进行归一化处理
    plotDataPoint(norXdata, ydata)#将数据点绘图
    preY = kNearestNeihgbor(norXdata[-1], norXdata[0:-1], ydata[0:-1],10)  #使用数据集中的最后一个样本作为测试数据，对应寻来你数据减少一个，k取值10
    #plt.plot(norXdata[-1][0], norXdata[-1][2], "ks" )
    #plt.show()
    #print("This example ",xdata[-1], "is predict to :", preY)
    print("This example {} is predict to {}".format(xdata[-1],preY ))
