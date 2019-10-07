import tensorflow as tf
import sklearn
import pickle
import numpy as np
from sklearn.metrics import accuracy_score 
cnn_fn = 'D:\PyDocument\\MyPaper\\seed28_model\\text_cnn.pkl'
rnn_fn = 'D:\PyDocument\\MyPaper\\seed28_model\\text_rnn.pkl'


sess = tf.InteractiveSession()
with open(cnn_fn, 'rb') as cnn_f:
    trscores, y_true = pickle.load(cnn_f)
with open(rnn_fn, 'rb') as rnn_f:
    tcscores = pickle.load(rnn_f)
print("end")

predictions = tf.argmax(tf.add(trscores, tcscores), 1).eval()
Accuracy = accuracy_score(y_true, predictions)
Precision =  sklearn.metrics.precision_score(y_true, predictions)
Recall =  sklearn.metrics.recall_score(y_true, predictions)
f1_score =  sklearn.metrics.f1_score(y_true, predictions)

print("accuracy", Accuracy)
print("Precision", Precision)
print("Recall", Recall)
print("f1_score", f1_score)

