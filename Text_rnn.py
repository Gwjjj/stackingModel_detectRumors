import tensorflow as tf
import pickle
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import random
import sklearn as sk
rumor_fn = 'D:\\PyDocument\\MyPaper\\rumor_vec.pkl'
norumor_fn = 'D:\\PyDocument\\MyPaper\\norumor_vec.pkl'
rueiddCId_fn = 'D:\\PyDocument\\MyPaper\\blog_pickle\\rumor_eid_childid.pkl'
norueiddCId_fn = 'D:\\PyDocument\\MyPaper\\blog_pickle\\norumor_eid_childid.pkl'

# 加载主博客子博客字典，加载是否谣言标签，加载doc2vec向量
def init_data():
    with open(rumor_fn, 'rb') as rumor_f:
        rumor_vec = pickle.load(rumor_f)
    with open(norumor_fn, 'rb') as norumor_f:
        norumor_vec = pickle.load(norumor_f)
    with open(rueiddCId_fn, 'rb') as rueiddCId_f:
        rueid_CId = pickle.load(rueiddCId_f)
    with open(norueiddCId_fn, 'rb') as norueiddCId_f:
        norueid_CId = pickle.load(norueiddCId_f)
    return rumor_vec, norumor_vec, rueid_CId, norueid_CId


rumor_vec, norumor_vec, rueid_CId_dict, norueid_CId_dict = init_data()
# count_rumor = len(rueid_CId)
# count_norumor = len(norueid_CId)
rueid_list = [reid for reid in rueid_CId_dict]
norueid_list = [nreid for nreid in norueid_CId_dict]
random.seed(28)
random.shuffle(rueid_list)
random.shuffle(norueid_list)
train_rueid_list =rueid_list[: -500]
train_norueid_list =norueid_list[: -500]
vaild_rueid_list =rueid_list[-500:]
vaild_norueid_list =norueid_list[-500:]
print('init end')
def get_train_label():
    input_list = []
    if random.randint(0,1):  # 谣言
        random_eid = random.choice(train_rueid_list)
        if len(rueid_CId_dict[random_eid]) < sequence_length:
            random_id = np.random.choice(rueid_CId_dict[random_eid], size=(sequence_length - 1 - len(rueid_CId_dict[random_eid])), replace=True)
            random_id = np.append(random_id,rueid_CId_dict[random_eid]) 
        else:
            random_id = random.sample(rueid_CId_dict[random_eid], sequence_length - 1)
        input_list.append(random_eid)
        input_list.extend(random_id)
        X_ = np.array([rumor_vec[x] for x in input_list])
        L_ = np.array([0, 1])
    else:
        random_eid = random.choice(train_norueid_list)
        if len(norueid_CId_dict[random_eid]) < sequence_length:
            random_id = np.random.choice(norueid_CId_dict[random_eid], size=(sequence_length - 1 - len(norueid_CId_dict[random_eid])), replace=True)
            random_id = np.append(random_id,norueid_CId_dict[random_eid]) 
        else:
            random_id = random.sample(norueid_CId_dict[random_eid], sequence_length - 1)
        input_list.append(random_eid)
        input_list.extend(random_id)
        X_ = np.array([norumor_vec[x] for x in input_list])
        L_ = np.array([1, 0])
    return X_, L_


def get_vaild_label(index, label):
    input_list = []
    if label:  # 谣言
        if len(rueid_CId_dict[index]) < sequence_length:
            random_id = np.random.choice(rueid_CId_dict[index], size=(sequence_length - 1 - len(rueid_CId_dict[index])), replace=True)
            random_id = np.append(random_id,rueid_CId_dict[index]) 
        else:
            random_id = random.sample(rueid_CId_dict[index], sequence_length - 1)
        input_list.append(index)
        input_list.extend(random_id)
        X_ = np.array([rumor_vec[x] for x in input_list])
        L_ = np.array([0, 1])
    else:
        if len(norueid_CId_dict[index]) < sequence_length:
            random_id = np.random.choice(norueid_CId_dict[index], size=(sequence_length - 1 - len(norueid_CId_dict[index])), replace=True)
            random_id = np.append(random_id,norueid_CId_dict[index]) 
        else:
            random_id = random.sample(norueid_CId_dict[index], sequence_length - 1)
        input_list.append(index)
        input_list.extend(random_id)
        X_ = np.array([norumor_vec[x] for x in input_list])
        L_ = np.array([1, 0])
    return X_, L_


embedding_dim = 768      
num_classes = 2           
num_layers= 2          
hidden_dim = 768
sequence_length = 60

x = tf.placeholder(tf.float32, [None, sequence_length, embedding_dim])
y_ = tf.placeholder(tf.float32, [None, 2])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
def gru_cell():  
    return tf.nn.rnn_cell.GRUCell(hidden_dim)

# 为每一个rnn核后面加一个dropout层
def dropout(): 
    cell = gru_cell()
    return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)


with tf.name_scope("rnn"):
    cells = [dropout() for _ in range(num_layers)]
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    # 堆叠了2层的RNN模型。
    _outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x, dtype=tf.float32)
    _outputs_re = tf.reshape(_outputs, [-1, sequence_length * hidden_dim])  
    # 取最后一个时序输出作为结果，也就是最后时刻和第2层的LSTM或GRU的隐状态。

with tf.name_scope("score"):
    # 全连接层，后面接dropout以及relu激活
    output_dropout = tf.nn.dropout(_outputs_re, dropout_keep_prob)
    W = tf.get_variable("W", shape=[sequence_length * hidden_dim, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(output_dropout, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")


with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_)
    loss = tf.reduce_mean(losses)

# Accuracy
with tf.name_scope("accuracy"):
    y_true = tf.argmax(y_, 1)
    correct_predictions = tf.equal(predictions, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

with tf.name_scope("optimize"):
    optim = tf.train.AdamOptimizer(1e-4).minimize(loss)


tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.8  #限制GPU内存占用率
sess = tf.Session(config=tfconfig)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
model_saver = "seed28_model/bert-more-textrnn"
ckpt_state = tf.train.get_checkpoint_state(model_saver)
saver = tf.train.Saver()
saver.restore(sess, ckpt_state.model_checkpoint_path)

x_va = []
y_va = []
for vaild_rueid in vaild_rueid_list:
    xv, yv = get_vaild_label(vaild_rueid, 1)
    x_va.append(xv)
    y_va.append(yv)
for novaild_rueid in vaild_norueid_list:
    xv, yv = get_vaild_label(novaild_rueid, 0)
    x_va.append(xv)
    y_va.append(yv)


acc_m = 0
for i in range(5000):
    X_train = []
    y = []
    for _ in range(64):
       xx, yy = get_train_label()
       X_train.append(xx)
       y.append(yy)
    sess.run(optim, feed_dict={x: X_train, y_: y, dropout_keep_prob: 0.7})
    if i % 10 == 0:
        acc = sess.run(accuracy, feed_dict={x: x_va, y_: y_va, dropout_keep_prob: 1})
        print(i, "iters: ", acc)


accuracy, predictions, y_true, scores = sess.run([accuracy, predictions, y_true, scores], feed_dict={x: x_va, y_: y_va, dropout_keep_prob: 1})


Precision =  sk.metrics.precision_score(y_true, predictions)
Recall =  sk.metrics.recall_score(y_true, predictions)
f1_score =  sk.metrics.f1_score(y_true, predictions)
print("accuracy", accuracy)
print("Precision", Precision)
print("Recall", Recall)
print("f1_score", f1_score)


write_text_rnn = open('seed28_model/text_rnn.pkl', 'wb')
pickle.dump(scores, write_text_rnn)

