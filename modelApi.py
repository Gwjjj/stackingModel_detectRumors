import tensorflow as tf
import pickle
import numpy as np
import random
import sklearn as sk
import time;  # 引入time模块
 

sequence_length = 60    
filter_sizes = [1, 3, 5]
num_filters = 64
num_classes = 2

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)


# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)




# x = tf.placeholder(tf.float32, [None, sequence_length, 768])
# x_rs = tf.reshape(x, [-1, sequence_length, 768, 1])
# y_ = tf.placeholder(tf.float32, [None, 2])
# dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

# x_pool = tf.nn.max_pool(
#             x_rs,
#             ksize=[1, 1, 768, 1], 
#             strides=[1, 1, 1, 1],
#             padding='VALID',
#             )
# x_pool_flat = tf.reshape(x_pool, [-1, sequence_length])
# # Create a convolution + maxpool layer for each filter size
# pooled_outputs = []
# for i, filter_size in enumerate(filter_sizes):
#     with tf.name_scope("conv-maxpool-%s" % filter_size):
#         # Convolution Layer
#         # filter_size 分别为3 4 5
#         filter_shape = [filter_size, 768, 1, num_filters]
#         W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
#         b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
#         conv = tf.nn.conv2d( # [None,56-3+1,1,128] [None,56-4+1,1,128] [None,56-5+1,1,128]
#             x_rs,
#             W,
#             strides=[1, 1, 1, 1],
#             padding="VALID",
#             name="conv")

#         # Apply nonlinearity
#         h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
#         # Maxpooling over the outputs
#         pooled = tf.nn.max_pool( 
#             h,
#             ksize=[1, sequence_length - filter_size + 1, 1, 1], 
#             strides=[1, 1, 1, 1],
#             padding='VALID',
#             name="pool")
#         print(pooled)
#         pooled_outputs.append(pooled)


# # Combine all the pooled features
# num_filters_total = num_filters * len(filter_sizes)
# h_pool = tf.concat(pooled_outputs, 3)
# h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
# final_pool = tf.concat([x_pool_flat, h_pool_flat], 1)
# num_filters_total += sequence_length
# # 全连接dropout
# with tf.name_scope("dropout"):
#     h_drop = tf.nn.dropout(final_pool, dropout_keep_prob)

# # Final (unnormalized) scores and predictions
# with tf.name_scope("output"):
#     W = tf.get_variable(
#         "W",
#         shape=[num_filters_total, num_classes],
#         initializer=tf.contrib.layers.xavier_initializer())
#     b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
#     scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
#     predictions = tf.argmax(scores, 1, name="predictions")

# # Calculate mean cross-entropy loss
# with tf.name_scope("loss"):
#     losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y_)
#     loss = tf.reduce_mean(losses)

# # Accuracy
# with tf.name_scope("accuracy"):
#     y_true = tf.argmax(y_, 1)
#     correct_predictions = tf.equal(predictions, y_true)
#     accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



# sess = tf.Session()
# print("init tf model")
# sess.run(tf.global_variables_initializer())



# # saver = tf.train.Saver(max_to_keep=1)
# model_saver = "api"



def restore(list):
    # with tf.Session() as sess:
    #     new_saver = tf.train.import_meta_graph('api/save_net.ckpt.meta')
    #     new_saver.restore(sess, 'my_net/save_net.ckpt')
    # print(list.shape) 
    # ckpt_state = tf.train.get_checkpoint_state(model_saver)
    # saver.restore(sess, ckpt_state.model_checkpoint_path)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('api_1/model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('api_1/'))
        detection_graph = tf.get_default_graph()  
        input_tensor = detection_graph.get_tensor_by_name('x:0')
        dropout_tensor = detection_graph.get_tensor_by_name('dropout_keep_prob:0')
        scores_emb = detection_graph.get_tensor_by_name("output/scores:0")
        print("sess run =========================")
        re = sess.run(scores_emb, feed_dict={input_tensor: list, dropout_tensor: 1})
        return re



# restore()