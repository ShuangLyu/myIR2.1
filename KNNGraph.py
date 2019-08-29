import  numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
import scipy.sparse as ss
from collections import deque
import heapq
from randomcube import RandomCube
import read_write

#构建图，并依据广度优先搜索查找出kbest个相似文档
class KNNG():

    def __init__(self, n_docs, k_best):
        self.graph = []
        self.n_docs = n_docs
        self.k_best = k_best

    def get_graph(self,path):
        self.graph = read_write.toload(path)

    def set_graph(self,path):
        read_write.tosave(path,self.graph)

    def buildgraph(self, docvec):
        d1 = docvec
        d2 = d1.T
        for i in range(d1.shape[0]):
            print(i)
            s = d1[i].dot(d2).toarray()
            self.graph.append(np.argsort(-s).tolist()[0][:self.k_best])

    def search(self,q,start_docsid,doc2vec):
        start_docs = ss.vstack(doc2vec[start_docsid]).tocsr()
        sim = q.dot(start_docs.T).toarray()
        simindex = np.argsort(-sim).tolist()[0] #返回相似矩阵对应位置索引下标
        tmp = [simindex[-1:][0] for x in range(0, self.k_best)]  # 保证堆内有kbest个元素,默认下标为相似矩阵最小值的下标
        tmp[:len(start_docsid)] = simindex[:self.k_best]  # 返回kbest个最相似文档的下标
        start_id = []
        for i in tmp:
            start_id.append((sim[0][i], start_docsid[i]))  # 堆默认以第一个元素排序
        # 用于标记是否被检查
        searched = set()
        searched.update(start_docsid)
        # 创建队列
        que = deque()
        # 创建堆,桶内最相似的kbest个文档的子节点入队
        heap = []
        for i in start_id:
            heapq.heappush(heap, i)
            que += self.graph[i[1]]
        while que:
            # 队列不为空，第一个相似文档出队
            doc = que.popleft()
            if doc not in searched:
                searched.add(doc)
                docvec = doc2vec[doc]
                simtmp = docvec.dot(q.T).toarray()
                if (simtmp[0], doc) >= heap[0]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (simtmp[0], doc))
                    que += self.graph[doc]
        result = heapq.nlargest(self.k_best, heap, key=lambda x: x[0])
        result = [i[1] for i in result]
        return result
