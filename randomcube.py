import pickle
import  numpy as np
import pandas as pd
import codecs
from scipy import sparse
from sklearn import preprocessing
import read_write

#类mylshash()用于查找与query落于同一个桶里的文档
class RandomCube():

    def __init__(self, n_bitcount, k_dim, rand = 'normal'):
        self.rng = np.random.RandomState(1234)
        self.w = self.rng.normal(0, 1, (n_bitcount,  k_dim)) #生成随机向量,随机生成0,1矩阵
        self.n_bitcount = n_bitcount
        self.middlenum = np.array([2**i for i in range(n_bitcount)])
        self.hashbucket = {}
        for i in range(2 ** n_bitcount): # 生成2**n_bitcount个桶
            self.hashbucket.setdefault(i, [])

    #求a * Wi.T, 将文档向量进行hash
    def hashsenvec(self,doc2vec):
        hash = doc2vec*self.w.T
        hashint = np.int64(hash>0)
        bucketid = hashint.dot(self.middlenum.T)
        return bucketid

    # 将所有文档划分到不同的桶中,并判断是否为空桶，若为空桶找到其临近桶
    def inbucket(self,doc2vec):
        bucketid = self.hashsenvec(doc2vec)
        for i,bucket in enumerate(bucketid):
            self.hashbucket[bucket].append(i)
        for j, v in self.hashbucket.items(): #判断是否为空桶，若为空桶找到其临近桶
            if len(v) == 0:
                m = self.n_bitcount-1
                while (m >= 0):
                    a = 1 << m
                    self.hashbucket[j].append(str(id(self.hashbucket[a^j])))
                    m = m - 1
        return self.hashbucket

    #根据query找到桶内相似文档
    def querydoc(self,q):
        qhash = self.hashsenvec(q)
        qhash = qhash[0]
        if isinstance(self.hashbucket[qhash][0], int):
            # print('这是非空桶')
            #print('桶里相似文档个数:%d' % (len(self.hashbucket[qhash])))
            return self.hashbucket[qhash]
        elif isinstance(self.hashbucket[qhash][0], str):
            try:
                # print('这是空桶')
                tmp = []
                for i in range(self.n_bitcount):  # 因为有n_bitcount个邻居
                    get_value = ctypes.cast(int(self.hashbucket[qhash][i]), ctypes.py_object).value  # 读取地址中的变量
                    if isinstance(get_value[0], int):
                        tmp.append(get_value)
                docid = []
                for i in tmp:
                    for j in i:
                        docid.append(j)
                #print('与邻居桶里相似文档的总个数:%d' % (len(docid)))
                return docid
            except:
                # print("无相似文本")
                return False
