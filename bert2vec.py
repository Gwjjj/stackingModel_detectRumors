from bert_serving.client import BertClient
import pickle
bc = BertClient(ip='localhost',check_version=False, check_length=False)
# bert-serving-start -model_dir D:\PyDocument\MyPaper\chinese_L-12_H-768_A-12 -num_worker=1
import numpy as np
with open('D:\\PyDocument\\MyPaper\\pdy\\yaoyantxt.pkl', 'rb') as norumor_docList_f:
    norumor_docList = pickle.load(norumor_docList_f)
print(len(norumor_docList))
# 读取微博，转化为句子向量

vec = bc.encode([s for s in norumor_docList[200000:]])
write_rumor_vec = open('yaoyan_vec'+ str(2) +'.pkl', 'wb')
pickle.dump(vec, write_rumor_vec)


# vec = bc.encode([s for s in norumor_docList[600000:]])
# write_rumor_vec = open('norumor_vec'+ str(4) +'.pkl', 'wb')
# pickle.dump(vec, write_rumor_vec)


    # for doc in rumor_docList[:5]:
    #     vec = bc.encode([doc])
    #     print(np.shape(vec))
        # rumor_vec.append(vec)


