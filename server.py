from flask import Flask, render_template, request

import datetime
import tensorflow as tf
import numpy as np

app = Flask(__name__)

X=tf.compat.v1.placeholder(tf.float32)
Y=tf.compat.v1.placeholder(tf.float32)
W= tf.Variable(tf.random.normal([1]), name="weight")
b= tf.Variable(tf.random.normal([1]),  name="bias")

hypothesis=W*X+b

saver = tf.compat.v1.train.Saver()
model = tf.compat.v1.global_variables_initializer()
sess=tf.compat.v1.Session()
sess.run(model)

saver=tf.compat.v1.train.Saver()
save_path=saver.save(sess,"./model/saved.cpkt")


@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        month = float(request.form['month'])
        mani = float(request.form['month'])

    price =0
    data=((month, mani), )
    arr = np.array(data, dtype=np.float32)

    x_data =arr[0]
    dict = sess.run(hypothesis, feed_dict={X:x_data})
    price = dict[0]

    return render_template('index.html', price=price)




if __name__ == '__main__':
    app.run(debug=True)
