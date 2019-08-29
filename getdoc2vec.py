from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import  numpy as np
import pandas as pd
import codecs
import pickle
from scipy import sparse
from sklearn import preprocessing
import read_write
from scipy.sparse import vstack

#类GetDoc2vec()形成词向量矩阵
class GetDoc2vec():

    def __init__(self,stopwordspath):
        self.stopwords = codecs.open(stopwordspath, 'r', encoding='GBK').readlines()
        self.stopwordlist = [w.strip() for w in self.stopwords]
        self.tfidfmodel = TfidfVectorizer()

    def get_model(self,path):
        self.tfidfmodel = read_write.toload(path)

    def set_model(self,path):
        read_write.tosave(path,self.tfidfmodel)

    def tokenization(self, doc):
        words = jieba.cut(doc)
        result = " ".join(words)
        return result

    def tomatrix(self, docs):
        dockeys = []
        if len(docs) > 1:
            for row in docs.itertuples():
                soup = getattr(row, 'GKXX')
                word_list = self.tokenization(soup)
                dockeys.append(word_list)
            self.tfidfmodel = TfidfVectorizer(stop_words = self.stopwordlist).fit(dockeys)
            sparse_result = self.tfidfmodel.transform(dockeys)
            return sparse_result

        else:   #将一条文档向量化
            word_list = self.tokenization(docs[0])
            dockeys.append(word_list)
            sparse_result = self.tfidfmodel.transform(dockeys)
            return sparse_result
