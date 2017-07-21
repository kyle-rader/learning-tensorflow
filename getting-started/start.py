import numpy as np
import tensorflow as tf

# TF Configuration
config = tf.ConfigProto(
    device_count = { 'gpu': 0 }
)
sess = tf.Session(config=config)

# Model Parameters
W = tf.Variable([1], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# Loss function - root mean squared
sqr_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(sqr_deltas)

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training Data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Init Variables
init = tf.global_variables_initializer()
sess.run(init)

# Training Loop
for i in range(100):
    sess.run(train, { x:x_train, y:y_train})

# Eval model accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
