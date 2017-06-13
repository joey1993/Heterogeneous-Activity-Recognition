import tensorflow as tf
import numpy as np


class HARCNN(object):
    """
    A CNN for activity classification.
    Uses a convolutional layer, followed by max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, filter_sizes, stride_size, filter_width, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.reshape_x = tf.reshape(self.input_x, [-1, 6, sequence_length/6])
        self.expand_x = tf.expand_dims(self.reshape_x, -1)
	print "expand_x: ",self.expand_x.get_shape()
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, filter_width, 1, num_filters]

                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.expand_x,
                    W,
                    strides=[1, 1, stride_size, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
		h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
		print "after conv: ", h.get_shape()
                # Maxpooling over the outputs
		pooled = tf.nn.max_pool(
		    h,
                    ksize=[1, 6-filter_size+1, sequence_length/(6*(filter_width-stride_size))-1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print "after pooled:", pooled.get_shape()
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        print "after concat diferent sizes of filters:", self.h_pool.get_shape()
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print "after flat:", self.h_pool_flat.get_shape()

        # Add dropout
#        with tf.name_scope("dropout"):
#	    self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_pool_flat, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
