import tensorflow as tf
import numpy as np
from pandas.io.parsers import read_csv

model=tf.compat.v1.global_variables_initializer()

data = read_csv('kf94.csv', sep=',')

xy= np.array(data, dtype=np.float32)

x_data = xy[:, [0]]
y_data = xy[:, [2]]

X=tf.compat.v1.placeholder(tf.float32)
Y=tf.compat.v1.placeholder(tf.float32)
W= tf.Variable(tf.random.normal([1]), name="weight")
b= tf.Variable(tf.random.normal([1]),  name="bias")

hypothesis=W*X+b 

cost = tf.reduce_mean(tf.square(hypothesis-Y))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000005)
train = optimizer.minimize(cost)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100001):
    cost_, hypo_, _=sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step%500 == 0:
        print("#", step, "손실비용: ", cost_)
        print("-마스크 가격:", hypo_[0])

saver=tf.compat.v1.train.Saver()
save_path=saver.save(sess,"./saved.cpkt")
print("학습된 모델을 저장했습니다.")

