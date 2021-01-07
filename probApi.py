# import pickle
# rueiddCId_fn = 'D:\\PyDocument\\MyPaper\\blog_pickle\\rumor_eid_childid.pkl'
# norueiddCId_fn = 'D:\\PyDocument\\MyPaper\\blog_pickle\\norumor_eid_childid.pkl'


# with open(norueiddCId_fn, 'rb') as norueiddCId_f:
#     norueid_CId = pickle.load(norueiddCId_f)
#     key = [eid for eid in norueid_CId]
#     print(len(key))
#     print(key[1132])
#     print(norueid_CId[373309])
#     # print(norueid_CId)

# 导入bert客户端
from bert_serving.client import BertClient
import modelApi as ma
import pickle
import numpy as np
import random
import math

sequence_length = 60    
filter_sizes = [1, 3, 5]
num_filters = 64



def choiceVec(arr):
    length = np.size(arr,0)
    if(length < sequence_length):
        zeors = np.zeros((sequence_length - length,768))
        arr = np.concatenate((arr,zeors),axis=0)
    elif(sequence_length < length):
        arr = arr[0:sequence_length,:]
    return arr


def softmax(inMatrix):
    """
    softmax计算公式函数
    :param inMatrix: 矩阵数据
    :return:
    """
    m,n = np.shape(inMatrix)  #得到m,n(行，列)
    outMatrix = np.mat(np.zeros((m,n)))  #mat生成数组
    soft_sum = 0
    for idx in range(0,n):
        outMatrix[0,idx] = math.exp(inMatrix[0,idx])  #求幂运算，取e为底的指数计算变成非负
        soft_sum +=outMatrix[0,idx]   #求和运算
    for idx in range(0,n):
        outMatrix[0,idx] = outMatrix[0,idx] /soft_sum #然后除以所有项之后进行归一化
    return outMatrix


def calProb(weiList):
    bc = BertClient()
    wei_vecs = bc.encode(weiList)
    vec = choiceVec(wei_vecs)
    vec = np.reshape(vec,(1,sequence_length,768))
    p = ma.restore(vec)
    print("softmax =============================")
    result = softmax(p)
    return result[0,1]

# bert-serving-start -model_dir D:\\PyDocument\\MyPaper\\chinese_L-12_H-768_A-12 -num_worker=4



def test1():
    df=open('D:\\PyDocument\\MyPaper\\testvec.pkl','rb')
    data3=pickle.load(df)   
    vec = choiceVec(data3)
    vec = choiceVec(vec)
    vec = np.reshape(vec,(1,sequence_length,768))
    print(vec)
    re = ma.restore(vec)
    print(re)


# test1()