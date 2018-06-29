# -*- coding: utf-8 -*-
# @Time    : 2018/6/28 10:53
# @Author  : quincyqiang
# @File    : regression.py
# @Software: PyCharm
import os
import tensorflow as tf
import input_data
import model
data=input_data.read_data_sets('MNIST_data',one_hot=True)

# 创建模型
with tf.variable_scope("regression"):
    x=tf.placeholder(tf.float32,[None,784])
    y,variables=model.regression(x)

# 训练
y_pred=tf.placeholder(tf.float32,[None,10])
cross_entropy=-tf.reduce_sum(y_pred*tf.log(y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

saver=tf.train.Saver(variables)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(1000):
        batch_x,batch_y=data.train.next_batch(100)
        sess.run(optimizer,feed_dict={x:batch_x,y_pred:batch_y})
    print(sess.run(accuracy,feed_dict={x:data.test.images,y_pred:data.test.labels}))

    model_path=os.path.join(os.path.dirname(__file__),'data','regression.ckpt')
    saver.save(sess,save_path=model_path)
    print("Saved:",model_path)
