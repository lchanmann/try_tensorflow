"""TensorFlow tutorial for ML Beginners at - https://www.tensorflow.org/tutorials/mnist/beginners/"""

# load mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

# 55000 data points in +mnist.train+
# 10000 data points in +mnist.test+
# 5000  data points in +mnist.validation+

# +mnist.train.images+ is a [55000 x 784] tensor
# an image is of 28x28 pixels
# features of an image is pixel intensity between [0, 1]

# +mnist.train.labels+ is a [55000 x 10] tensor
# label, a number between 0 and 9, corresponds to a image in the training set
# label is encoded as one hot eg. [0,0,0,0,0,0,0,0,0,1] = 9

# using TensorFlow high-level API for python
import tensorflow as tf

# create a 2-D tensor of floating-point number with variable rows and 784 columns placeholder for input
x = tf.placeholder(tf.float32, [None, 784])

# initialize modifiable 2-D tensor for Weights and Biases as zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# compute the model prediction
y = tf.nn.softmax(tf.matmul(x, W) + b)

# create place holder for label
y_prime = tf.placeholder(tf.float32, [None, 10])

# compute cross entropy loss
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_prime * tf.log(y), reduction_indices=[1]) )
# preferably compute combine softmax cross entropy to reduce numberical stability
cross_entropy = tf.nn.softmax_cross_entropy_with_logits( tf.matmul(x, W) + b, y_prime )

# set up the training step to use Gradient Descent algorithm
learning_rate = 0.5
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# initialize the variable created with +tf.Variable()+
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# stochastically train the model(Weights and Biases) for 1000 steps
batch_size = 100
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict = {x: batch_xs, y_prime: batch_ys})

# evaluate the model by comparing with the ground truth
y_hat = tf.argmax(y, 1)
y_truth = tf.argmax(y_prime, 1)

correct_prediction = tf.equal(y_hat, y_truth)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run the model on the test set
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y_prime: mnist.test.labels}))
