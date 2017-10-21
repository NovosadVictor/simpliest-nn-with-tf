import tensorflow as tf
from random import random

learning_rate = 0.01
weights = tf.Variable([[2 * random() - 1 for j in range(2)] for i in range(3)], dtype=tf.float32)
biases = tf.Variable([2 * random() - 1, 2 * random() - 1], dtype=tf.float32)

input = tf.placeholder(tf.float32)
output = tf.placeholder(tf.float32)

result = tf.add(tf.matmul(input, weights), biases)

loss = tf.reduce_mean(tf.square(tf.add(output, -(result))))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(loss)

x_train = [[0, 0, 1], [1, 0, 1], [1, 1, 0], [0, 1, 0], [0, 0, 0]]
y_labels = [[1, 0], [0, 1], [0, 1], [1, 0], [[1, 0]]]

sess = tf.Session()

for i in range(len(x_train)):
    reshape_x = tf.reshape(x_train[i], shape=[1, 3])
    reshape_y = tf.reshape(y_labels[i], shape=[1, 2])
    x, y = sess.run([reshape_x, reshape_y])
    x_train[i] = x
    y_labels[i] = y

init = tf.global_variables_initializer()

sess.run(init)

for e in range(1000):
    for i in range(len(x_train)):
        print(i, e)
        sess.run(train, {input: x_train[i], output: y_labels[i]})

print("Error on training set: ")
for i in range(len(x_train)):
    answer, error = sess.run([result, loss], {input: x_train[i], output: y_labels[i]})
    print("Answer, Loss is: ", answer, error)

print("Error on other examples: ")
x_test = [[0, 1, 1], [1, 0, 0], [1, 1, 1]]
y_test = [[1, 0], [0, 1], [0, 1]]
for i in range(len(x_test)):
    reshape_x = tf.reshape(x_test[i], shape=[1, 3])
    reshape_y = tf.reshape(y_test[i], shape=[1, 2])
    x, y = sess.run([reshape_x, reshape_y])
    x_test[i] = x
    y_test[i] = y

for i in range(len(x_test)):
    answer, error = sess.run([result, loss], {input: x_test[i], output: y_test[i]})
    print("Answer, Loss is: ", answer, error)



