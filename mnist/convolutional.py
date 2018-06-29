import os
import model
import tensorflow as tf
import input_data

data=input_data.read_data_sets('MNIST_data',one_hot=True)

# model
with tf.variable_scope('convolutional'):
    x=tf.placeholder(tf.float32,[None,784],name='x')
    keep_prob=tf.placeholder(tf.float32)
    y,variables=model.convolutional(x,keep_prob)

# train
y_pred=tf.placeholder(tf.float32,[None,10],name='y_pred')

cross_entropy=-tf.reduce_mean(y_pred*tf.log(y))
optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_pred,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


saver=tf.train.Saver(variables)
with tf.Session() as sess:
    merged_summary_op=tf.summary.merge_all()
    summary_writer=tf.summary.FileWriter('/tmp/mnist_log/1',sess.graph)
    summary_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())

    for i in range(5400):
        batch=data.train.next_batch(50)

        if i%100==0:
            train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_pred:batch[1],keep_prob:1.0})
            print("step %d,training accuracy %g" % (i,train_accuracy))

        sess.run(optimizer,feed_dict={x:batch[0],y_pred:batch[1],keep_prob:0.5})

    print(sess.run(accuracy, feed_dict={x: data.test.images, y_pred: data.test.labels, keep_prob: 1.0}))

    model_path=os.path.join(os.path.dirname(__file__),'data','convolutional.ckpt')
    saver.save(sess,model_path)
    print("Saved:", model_path)