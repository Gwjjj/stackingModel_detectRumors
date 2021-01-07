import jieba
import gensim
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pickle


 
def ceshi(str1):
    model_dm = Doc2Vec.load("D:\\PyDocument\\MyPaper\\Doc2vec\\mydoc.model")
    ##此处需要读入你所需要进行提取出句子向量的文本   此处代码需要自己稍加修改一下
    ##你需要进行得到句子向量的文本，如果是分好词的，则不需要再调用结巴分词
    test_text = str(''.join(jieba.cut(str1)).encode('utf-8')).split(' ')
 
    inferred_vector_dm = model_dm.infer_vector(test_text) ##得到文本的向量
    print(inferred_vector_dm)
    return inferred_vector_dm

with open('D:\\PyDocument\\MyPaper\\pdy\\yaoyantxt.pkl', 'rb') as norumor_docList_f:
    norumor_docList = pickle.load(norumor_docList_f)
print(len(norumor_docList))
# 读取微博，转化为句子向量
vec = []
for i in norumor_docList[0:5]:
    vec.append(ceshi(norumor_docList[i]))
vecs = np.array(vec)
with open('D:\\PyDocument\\MyPaper\\Doc2vec\\rumor'+ str(1) +'.pkl', 'wb') as write_rumor_vec:
    pickle.dump(vec, write_rumor_vec)