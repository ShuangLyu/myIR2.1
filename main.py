from getdoc2vec import GetDoc2vec
from randomcube import RandomCube
from KNNGraph import KNNG
import pandas as pd
import read_write
import time

def get_nbit(docvec,kbest):
    l = docvec.shape[0]/kbest
    nbit = 0
    while True:
        if 2**nbit >= l:
            return nbit-1
        else:
            nbit += 1

def rukou(k_best,qdoc,doc,doc2vec):
    time1 = time.time()
    queryvec = GetDoc2vec(stopwordspath = 'D:/xfdata/stop_words.txt')
    queryvec.get_model(path = "D:/xftest/forir/tfidf_model.pkl")
    q = queryvec.tomatrix(docs = qdoc)
    time2 = time.time()
    print("生成词向量所需时间: "+str(time2-time1))
    n_bit = get_nbit(doc2vec, k_best)
    rdc = RandomCube(n_bitcount = n_bit, k_dim = doc2vec.shape[1])
    rdc.inbucket(doc2vec)
    start_docids = rdc.querydoc(q)
    time3 = time.time()
    print("生成搜索起点所需时间："+str(time3-time2))
    if start_docids is not False:
        knng = KNNG(n_docs = doc2vec.shape[0], k_best = k_best)
        # knng.buildgraph(doc2vec)
        # knng.set_graph("D:/xftest/forir/graph.pkl")
        knng.get_graph("D:/xftest/forir/graph1.pkl")
        similarid = knng.search(q, start_docids, doc2vec)
        print(doc[similarid])
    else:
        print("找不到相似内容")
    time4 = time.time()
    print("检索query所需时间："+str(time4-time3))

if __name__ == "__main__":
    stime = time.time()
    q = ['反应邻居养猪，污水乱排放，很脏很臭，希望得到查处']
    rukou(k_best =50,
          qdoc = q,
          doc = pd.read_csv('D:/xftest/gkxx.csv', ',', encoding = 'ansi', header = 0)['GKXX'].astype(str),
          doc2vec = read_write.toload("D:/xftest/forir/doc2vec.pkl"))
    print("全部用时："+str(time.time()-stime))

