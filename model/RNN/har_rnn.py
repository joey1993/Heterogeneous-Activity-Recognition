import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np



class HARRNN(object):
    """
    A CNN for activity classification.
    Uses a convolutional layer, followed by max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, n_hidden, num_classes, l2_reg_lambda=0.0):

        self.n_hidden = n_hidden
        self.n_steps = sequence_length / 6
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.reshape_x = tf.reshape(self.input_x, [-1, 6, sequence_length/6])
        self.transpose_x = tf.transpose(self.reshape_x, perm=[0, 2, 1])
        print "adjusted input shape: ", self.transpose_x.get_shape()
        self.unstacked_x = tf.unstack(self.transpose_x, self.n_steps, axis=1)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("lstm") :
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
            #lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.dropout_keep_prob)
            outputs, states = tf.nn.rnn(lstm_cell, self.unstacked_x, dtype=tf.float32)
            self.lstm_output = outputs[-1]

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.n_hidden, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.lstm_output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
