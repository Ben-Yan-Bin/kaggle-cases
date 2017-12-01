from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.4)

encoder_train = OneHotEncoder()
Y_train = np.transpose([y_train])
Y_train = encoder_train.fit_transform(Y_train).toarray()

# print(Y_train)

encoder_test = OneHotEncoder()
Y_test = np.transpose([y_test])
Y_test = encoder_test.fit_transform(Y_test).toarray()

# print(X_test)

# Parameters
learning_rate = 0.001
training_epochs = 3000
# batch_size = 30
display_step = 100

# Define how many inputs and outputs are in our neural network
number_of_inputs = 4
number_of_outputs = 3

# Define how many neurons we want in each layer of our neural network
layer_1_nodes = 100
layer_2_nodes = 100
layer_3_nodes = 50

w_init = tf.random_normal_initializer
b_init = tf.random_normal_initializer

dropout_rate = 0.6

# Input Layer
with tf.variable_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))

with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name='weights1', shape=[number_of_inputs, layer_1_nodes],
                              initializer=w_init)
    biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=b_init)
    w_b = tf.add(tf.matmul(X, weights), biases)
    w_b = tf.nn.dropout(w_b, dropout_rate)
    layer_1_output = tf.nn.relu(w_b)

# Layer 2
with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name='weights2', shape=[layer_1_nodes, layer_2_nodes],
                              initializer=w_init)
    biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=b_init)
    w_b = tf.add(tf.matmul(layer_1_output, weights), biases)
    w_b = tf.nn.dropout(w_b, dropout_rate)
    layer_2_output = tf.nn.relu(w_b)
    # layer_2_output = tf.nn.relu(tf.add(tf.matmul(layer_1_output, weights), biases))

# # Layer 3
# with tf.variable_scope('layer_3'):
#     weights = tf.get_variable(name='weights3', shape=[layer_2_nodes, layer_3_nodes],
#                               initializer=w_init)
#     biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=b_init)
#     w_b = tf.add(tf.matmul(layer_2_output, weights), biases)
#     w_b = tf.nn.dropout(w_b, dropout_rate)
#     layer_3_output = tf.nn.relu(w_b)
#     # layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Output Layer
with tf.variable_scope('output'):
    weights = tf.get_variable(name='weights4', shape=[layer_2_nodes, number_of_outputs],
                              initializer=w_init)
    biases = tf.get_variable(name='biases4', shape=[number_of_outputs], initializer=b_init)
    prediction = tf.matmul(layer_2_output, weights) + biases

# Section Two: Define the cost function of the neural network
with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, number_of_outputs))
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=Y))

# Section Three: Define the optimizer function
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary operation to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

# Initialize a session so that we can run Titanic_TensorFlow operations
with tf.Session() as sess:
    # Run the global variable initializer to initialize all variables and layers
    sess.run(tf.global_variables_initializer())
    training_writer = tf.summary.FileWriter('./logs/training', sess.graph)
    testing_writer = tf.summary.FileWriter('./logs/testing', sess.graph)

    for epoch in range(training_epochs):
        # Feed in the training data and do one step of neural network training
        sess.run(optimizer, feed_dict={X: X_train, Y: Y_train})

        if epoch % display_step == 0:
            # Print the current training status to the screen
            training_cost, train_summary = sess.run([cost, summary], feed_dict={X: X_train, Y: Y_train})
            test_cost, test_summary = sess.run([cost, summary], feed_dict={X: X_test, Y: Y_test})
            # test_prediction = sess.run([prediction], feed_dict={X: X_test, Y: Y_test})

            training_writer.add_summary(train_summary, epoch)
            testing_writer.add_summary(test_summary, epoch)

            print('Training pass: {}'.format(epoch))
            print('Train Cost is {}, Test Cost is {}'.format(training_cost, test_cost))
            # print('Prediction: ', tf.nn.softmax(prediction).eval(feed_dict={X: X_test, Y: Y_test}))

    # Training is now complete
    print('Training is complete!')

    final_training_cost = sess.run(cost, feed_dict={X: X_train, Y: Y_train})
    final_test_cost = sess.run(cost, feed_dict={X: X_test, Y: Y_test})
    # final_prediction = sess.run(prediction, feed_dict={X: X_train, Y: Y_train})

    print('Final Training Cost is {}, Final Test Cost is {}'.format(final_training_cost, final_test_cost))
    # print('Final prediction:', final_prediction)
    # print(type(final_prediction))

    # Test model
    pred = tf.nn.softmax(prediction)  # Apply softmax to logits
    # print('PRED', pred.eval(feed_dict={X: X_test, Y: Y_test}))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
    print("Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
