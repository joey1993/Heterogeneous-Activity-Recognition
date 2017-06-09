#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from har_cnn import HARCNN
from tensorflow.contrib import learn
import yaml
from sklearn import metrics

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .0005, "Percentage of the training data to use for validation")

# Model Hyperparameters\
tf.flags.DEFINE_string("filter_sizes", "4,5,6", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("stride_size", 5, "Size of filter stride (defaut: 5)")
tf.flags.DEFINE_integer("filter_width", 10, "size of time window concluded by one filter (default: 10)")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.01, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

dataset_name = cfg["datasets"]["default"]

# Data Preparation
# ==================================================

# Load data
print("Loading data...Cool Joey :-)")
datasets = None

if dataset_name == "HAR_small":
    datasets_train = data_helpers.get_datasets(cfg["datasets"][dataset_name]["training_data_file"]["path"])
    datasets_test = data_helpers.get_datasets(cfg["datasets"][dataset_name]["testing_data_file"]["path"])

x, y = data_helpers.load_data_labels(datasets_train)
x_test, y_test = data_helpers.load_data_labels(datasets_test)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/dev set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("Train/Test split: {:d}/{:d}".format(len(y_train), len(y_test)))

# Training
# ==================================================

with tf.Graph().as_default():
    tf.set_random_seed(100)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    print "sequence_length, ",x_train.shape[1]
    print "num_classes, ",y_train.shape[1]
    with sess.as_default():
        harcnn = HARCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
	    stride_size=FLAGS.stride_size,
	    filter_width = FLAGS.filter_width,
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(harcnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.pardir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", harcnn.loss)
        acc_summary = tf.summary.scalar("accuracy", harcnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              harcnn.input_x: x_batch,
              harcnn.input_y: y_batch,
              harcnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, harcnn.loss, harcnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % 100 == 0: print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, test=False, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              harcnn.input_x: x_batch,
              harcnn.input_y: y_batch,
              harcnn.dropout_keep_prob: 1
            }
            step, summaries, loss, accuracy, y_pred, input_ys = sess.run(
                [global_step, dev_summary_op, harcnn.loss, harcnn.accuracy, harcnn.predictions, harcnn.input_y],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

            y_true = np.argmax(input_ys,1)

            print "Precision", metrics.precision_score(y_true, y_pred, average='weighted')
            print "Recall", metrics.recall_score(y_true, y_pred, average='weighted')
            print "F1_score", metrics.f1_score(y_true, y_pred, average='weighted') 
            print metrics.classification_report(y_true, y_pred, target_names=datasets_test['target_names'])
	    print "confusion_matrix"
            print metrics.confusion_matrix(y_true, y_pred)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
	    x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                #dev_step(x_dev, y_dev, writer=dev_summary_writer)
                dev_step(x_test, y_test, True, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
