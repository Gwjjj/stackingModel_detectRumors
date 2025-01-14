import tensorflow as tf
import pickle
import numpy as np
import random
import sklearn as sk
import time;  # 引入time模块
 

rumor_fn = 'D:\\PyDocument\\MyPaper\\bert\\rumor_vec.pkl'
norumor_fn = 'D:\\PyDocument\\MyPaper\\bert\\norumor_vec.pkl'
rueiddCId_fn = 'D:\\PyDocument\\MyPaper\\blog_pickle\\rumor_eid_childid.pkl'
norueiddCId_fn = 'D:\\PyDocument\\MyPaper\\blog_pickle\\norumor_eid_childid.pkl'
sequence_length = 60    
filter_sizes = [1, 3, 5]
fisrt_conv = 5
first_num_filters = 8
num_filters = 32
first_pool_len = 5

# 加载主博客子博客字典，加载是否谣言标签，加载bert2vec向量
def init_data():
    with open(rumor_fn, 'rb') as rumor_f:
        rumor_vec = pickle.load(rumor_f)
    with open(norumor_fn, 'rb') as norumor_f:
        norumor_vec = pickle.load(norumor_f)
    with open(rueiddCId_fn, 'rb') as rueiddCId_f:
        rueid_CId = pickle.load(rueiddCId_f)
    with open(norueiddCId_fn, 'rb') as norueiddCId_f:
        norueid_CId = pickle.load(norueiddCId_f)
    print("init end")
    return rumor_vec, norumor_vec, rueid_CId, norueid_CId


rumor_vec, norumor_vec, rueid_CId_dict, norueid_CId_dict = init_data()
# count_rumor = len(rueid_CId)
# count_norumor = len(norueid_CId)
rueid_list = [reid for reid in rueid_CId_dict]
norueid_list = [nreid for nreid in norueid_CId_dict]

zeors = np.zeros(768)
rumor_vec.append(zeors)
norumor_vec.append(zeors)
zero_ru = len(rumor_vec) - 1
zero_noru = len(norumor_vec) - 1
random.shuffle(rueid_list)
random.shuffle(norueid_list)
train_rueid_list =rueid_list[: -400]
train_norueid_list =norueid_list[: -400]
vaild_rueid_list =rueid_list[-400:]
vaild_norueid_list =norueid_list[-400:]

num_classes = 2
def get_train_label():
    input_list = []
    if random.randint(0,1):  # 谣言
        random_eid = random.choice(train_rueid_list)
        if len(rueid_CId_dict[random_eid]) < sequence_length:
            random_id = rueid_CId_dict[random_eid]
            for _ in range(sequence_length - len(rueid_CId_dict[random_eid]) - 1):
                random_id.append(zero_ru)
        else:
            random_id = random.sample(rueid_CId_dict[random_eid], sequence_length - 1)
        input_list.append(random_eid)
        input_list.extend(random_id)
        X_ = np.array([rumor_vec[x] for x in input_list])
        L_ = np.array([0, 1])
    else:
        random_eid = random.choice(train_norueid_list)
        if len(norueid_CId_dict[random_eid]) < sequence_length:
            random_id = norueid_CId_dict[random_eid]
            for _ in range(sequence_length - len(norueid_CId_dict[random_eid]) - 1):
                random_id.append(zero_noru)
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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# def conv2d(x, W):
#     return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='VALID')


# def max_pool(x):
#     return tf.nn.max_pool(x, ksize=[1, sequence_length-filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')


# l2_loss = tf.constant(0.0)

x = tf.placeholder(tf.float32, [None, sequence_length, 768], name="x")
x_rs = tf.reshape(x, [-1, sequence_length, 768, 1])
y_ = tf.placeholder(tf.float32, [None, 2])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# x_pool = tf.nn.max_pool(
#             x_rs,
#             ksize=[1, 1, 768, 1], 
#             strides=[1, 1, 1, 1],
#             padding='VALID',
#             )
# x_pool_flat = tf.reshape(x_pool, [-1, sequence_length])
# Create a convolution + maxpool layer for each filter size
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        # filter_size 分别为3 4 5
        filter_shape = [filter_size, 768, 1, num_filters]
        if i == 1:
            first_pooled = tf.nn.max_pool( 
                x_rs,
                ksize=[1, 1, first_pool_len, 1], 
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="first_pool")
            filter_shape_1 = [filter_size, 768-first_pool_len+1, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d( # [None,56-3+1,1,128] [None,56-4+1,1,128] [None,56-5+1,1,128]
                first_pooled,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool( 
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1], 
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            print(pooled)
            pooled_outputs.append(pooled)
        else:
            filter_shape_2 = [1, fisrt_conv, 1, first_num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape_2, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[first_num_filters]), name="b")
            conv = tf.nn.conv2d( 
                x_rs,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            filter_shape_2_2 = [filter_size, 768-fisrt_conv+1, first_num_filters, num_filters]
            W1 = tf.Variable(tf.truncated_normal(filter_shape_2_2, stddev=0.1), name="W1")
            b1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b1")
            conv_1 = tf.nn.conv2d( 
                h,
                W1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h_1 = tf.nn.relu(tf.nn.bias_add(conv_1, b1), name="relu_1")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool( 
                h_1,
                ksize=[1, sequence_length - filter_size + 1, 1, 1], 
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            print(pooled)
            pooled_outputs.append(pooled)
# Combine all the pooled features
num_filters_total = num_filters * 3
# len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
# final_pool = tf.concat([x_pool_flat, h_pool_flat], 1)
# num_filters_total += sequence_length
# 全连接dropout
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# Final (unnormalized) scores and predictions
with tf.name_scope("output"):
    W = tf.get_variable(
        "W",
        shape=[num_filters_total, num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")

# Calculate mean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_)
    loss = tf.reduce_mean(losses)

# Accuracy
with tf.name_scope("accuracy"):
    y_true = tf.argmax(y_, 1)
    correct_predictions = tf.equal(predictions, y_true)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())



# saver = tf.train.Saver(max_to_keep=1)
# model_saver = "seed28_model/bert-more-textcnn"

# ckpt_state = tf.train.get_checkpoint_state(model_saver)
# saver = tf.train.Saver()
# saver.restore(sess, ckpt_state.model_checkpoint_path)
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


# 训练过程
# acc_m = 0
for i in range(50000):
    X_train = []
    y = []
    for _ in range(64):
       xx, yy = get_train_label()
       X_train.append(xx)
       y.append(yy)
    sess.run(train_step, feed_dict={x: X_train, y_: y, dropout_keep_prob: 0.5})
    if i % 500 == 0:
        acc = sess.run(accuracy, feed_dict={x: x_va, y_: y_va, dropout_keep_prob: 1})
        print(i, "iters: ", acc)
        localtime = time.asctime( time.localtime(time.time()) )
        print("本地时间为 :", localtime)
        if 0.950 < acc < 0.97:
            break

saver = tf.train.Saver(max_to_keep=1)
model_save_path = "cnn_1/model"
saver.save(sess=sess, save_path=model_save_path)
# ckpt_state = tf.train.get_checkpoint_state(model_saver)
# saver = tf.train.Saver()
# saver.restore(sess, ckpt_state.model_checkpoint_path)



accuracy, predictions, y_true, scores = sess.run([accuracy, predictions, y_true, scores], feed_dict={x: x_va, y_: y_va, dropout_keep_prob: 1.0})


Precision =  sk.metrics.precision_score(y_true, predictions)
Recall =  sk.metrics.recall_score(y_true, predictions)
f1_score =  sk.metrics.f1_score(y_true, predictions)
print("accuracy", accuracy)
print("Precision", Precision)
print("Recall", Recall)
print("f1_score", f1_score)


# write_text_cnn = open('seed28_model/text_cnn.pkl', 'wb')
# pickle.dump([scores,y_true], write_text_cnn)