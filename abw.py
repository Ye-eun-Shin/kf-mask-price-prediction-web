import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

X=tf.compat.v1.placeholder(tf.float32)
Y=tf.compat.v1.placeholder(tf.float32)
W= tf.Variable(tf.random.normal([1]), name="weight")
b= tf.Variable(tf.random.normal([1]),  name="bias")

hypothesis=W*X+b 

saver=tf.compat.v1.train.Saver()
model=tf.compat.v1.global_variables_initializer()
month=float(input('몇월?: '))
mani=float(input('생산량?: '))

with tf.compat.v1.Session() as sess:
    sess.run(model)

    save_path="./saved.cpkt"
    saver.restore(sess,save_path)

    data=((month, mani), )
    arr = np.array(data, dtype=np.float32)

    x_data =arr[0]
    dict = sess.run(hypothesis, feed_dict={X:x_data})
    print(dict[0])
