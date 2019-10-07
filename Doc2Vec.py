import gensim
from gensim.models.doc2vec import Doc2Vec
import pickle
from gensim.models.doc2vec import TaggedDocument

def init_data():
    with open('labelAndCId.pkl', 'rb') as labelAndCId_f:
        eid_labelAndCId_dict = pickle.load(labelAndCId_f)
    with open('docList.pkl', 'rb') as docList_f:
        doc_list = pickle.load(docList_f)
    return eid_labelAndCId_dict, doc_list

eid_labelAndCId_dict, doc_list = init_data()


def X_train():
    x_train = []
    for i, cut in enumerate(doc_list):
        document = TaggedDocument(cut, [i])
        x_train.append(document)
    return x_train


def train(xtrain, size = 300):
    # gensim Doc2Vec 提供了 DM 和 DBOW 两个模型。gensim 的说明文档建议多次训练数据集并调整学习速率或在每次训练中打乱输入信息的顺序以求获得最佳效果。
    model = Doc2Vec(xtrain, min_count=5, window=8, vector_size=size, sample=1e-3, negative=5, workers=4)
    model.train(xtrain, total_examples=model.corpus_count, epochs=10)
    model.save('doc2vec.model')  # 保存模型训练结果
    return model



train(X_train())