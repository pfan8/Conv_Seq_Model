import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
from bleu import evaluate
from tensorflow.python import debug as tf_debug
import pdb

class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - labels: labels of shape (400000, 17) 
                - idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.data = data
        self.val_data = val_data
        self.n_epochs = kwargs.pop('n_epochs', 10)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def train(self):
        # train/val dataset
        n_examples = self.data['labels'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        features = self.data['features']
        labels = self.data['labels']
        idxs = np.arange(features.shape[0])

        # build graphs for training model and sampling labels
        loss = self.model.build_model()

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            gvs = optimizer.compute_gradients(loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs)

        tf.get_variable_scope().reuse_variables()
        # Metrics and F1
        max_len = len(labels[0])
        generated_labels = self.model.build_sampler(max_len)

        # metric_add_result = self.model.labels + generated_labels
        # equal_to_0 = tf.equal(metric_add_result, 0)
        # equal_to_2 = tf.equal(metric_add_result, 2)
        # true_negative = tf.reduce_sum(tf.cast(equal_to_0, tf.int32))
        # true_positive = tf.reduce_sum(tf.cast(equal_to_2, tf.int32))
        metric_minus_result = self.model.labels - generated_labels
        equal_to_0 = tf.equal(metric_minus_result, 0)
        accurancy = tf.reduce_sum(tf.cast(equal_to_0, tf.float32))
        # equal_to_1 = tf.equal(metric_minus_result, 1)
        # equal_to_n1 = tf.equal(metric_minus_result, -1)
        # false_negative = tf.reduce_sum(tf.cast(equal_to_1, tf.int32))
        # false_positive = tf.reduce_sum(tf.cast(equal_to_n1, tf.int32))
        
        # summary op
        tf.summary.scalar('accuracy',accurancy)
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in gvs:
            tf.summary.histogram(var.op.name + '/gradient', grad)

        summary_op = tf.summary.merge_all()
        


        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # tf.global_variables_initializer()
            tf.initialize_all_variables().run()
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=40)

            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            prev_acc = -1
            curr_acc = 0
            prev_test_acc = -1
            curr_test_acc = 0
            max_acc = 0.0
            start_t = time.time()

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                labels = labels[rand_idxs]
                idxs = idxs[rand_idxs]
                # Metrics
                # TP = 0
                # FP = 0
                # TN = 0
                # FN = 0

                for i in range(n_iters_per_epoch):
                    labels_batch = labels[i * self.batch_size:(i + 1) * self.batch_size]
                    idxs_batch = idxs[i * self.batch_size:(i + 1) * self.batch_size]
                    features_batch = features[idxs_batch]
                    feed_dict = {self.model.features: features_batch, self.model.labels: labels_batch}
                    _, l, acc = sess.run([train_op, loss, accurancy], feed_dict)
                    # tp,fp,tn,fn = sess.run([true_positive, false_positive,\
                    #                         true_negative, false_negative], feed_dict)
                    # _, l, debug_var = sess.run([train_op, loss, debug_var], feed_dict)
                    # print "feature_batch:%s" % features_batch[0]
                    # print "labels_batch:%s" % labels_batch[0]
                    # print "slengths_batch:%s" % slengths_batch[0]
                    # print "debug_var:{},{},{}".format(debug_var[0].shape,debug_var[1],debug_var[2])
                    curr_loss += l
                    curr_acc += acc
                    # TP += tp
                    # FP += fp
                    # TN += tn
                    # FN += fn

                    # write summary for tensorboard visualization
                    if i % 10 == 0:
                        summary = sess.run(summary_op, feed_dict)
                        summary_writer.add_summary(summary, e * n_iters_per_epoch + i)

                    if (i + 1) % self.print_every == 0:
                        print "\nTrain loss at epoch %d & iteration %d (mini-batch): %.5f" % (e + 1, i + 1, l)
                        ground_truths = labels[idxs == idxs_batch[0]]
                        # decoded = decode_labels(ground_truths, self.model.idx_to_word)
                        decoded = ground_truths
                        for j, gt in enumerate(decoded):
                            print "Ground truth %d: %s" % (j + 1, gt)
                        gen_caps = sess.run(generated_labels, feed_dict)
                        # decoded = decode_labels(gen_caps, self.model.idx_to_word)
                        decoded = gen_caps
                        print "Result: %s\n" % decoded[0]
                
                print "Previous epoch train loss: ", prev_loss
                print "Current epoch train loss: ", curr_loss
                # assert (TP+FP+TN+FN) == n_examples * max_len
                # curr_acc = (TP + TN) / float(n_examples * max_len)
                curr_acc /= float(n_examples * max_len)
                print "Previous epoch train acc: ", prev_acc
                print "Current epoch train acc: ", curr_acc
                feed_dict_test = {self.model.features: self.val_data['features'], self.model.labels: self.val_data['labels']}
                curr_test_acc = sess.run(accurancy, feed_dict_test)
                curr_test_acc /= float(n_examples * max_len)
                print "Previous epoch test acc: ", prev_test_acc
                print "Current epoch test acc: ", curr_test_acc
                # print "True Positive: ", TP
                # print "False Positive: ", FP
                # print "True Negative: ", TN
                # print "False Negative: ", FN
                # recall = TP / float(TP + FN )
                # print "Recall:", recall
                # F1 = curr_acc * recall * 2 / (curr_acc + recall)
                # print "F1:", F1
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                prev_acc = curr_acc
                prev_test_acc = curr_test_acc
                curr_loss = 0

                # save model's parameters
                if max_acc < curr_test_acc:
                    max_acc= curr_test_acc
                    saver.save(sess, os.path.join(self.model_path, 'model'), global_step=e + 1)
                    print "model of %.2f acc saved." % (max_acc)

                curr_acc = 0
                curr_test_acc = 0

    def test(self, data, split='train', attention_visualization=False, save_sampled_labels=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - labels: labels of shape (24210, 17)
            - idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_labels: Mapping feature to labels (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_labels: If True, save sampled labels to pkl file for computing BLEU scores.
        '''

        features = data['features']
        labels = data['labels']
        n_examples = data['labels'].shape[0]
        n_iters_per_epoch = int(np.ceil(float(n_examples) / self.batch_size))
        # build a graph to sample labels
        max_len = len(labels[0])
        sampled_labels = self.model.build_sampler(max_len)  # (N, max_len, L)
        tf.get_variable_scope().reuse_variables()
        rank5_results = self.model.get_rank5_result(max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        # tf.get_variable_scope().reuse_variables()
        # loss, debug_var = self.model.build_model()
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)

            all_gen_cap = np.ndarray((labels.shape[0], max_len))
            all_rank5_results = np.ndarray((labels.shape[0], max_len, 5))
            start_t = time.time()
            for i in range(n_iters_per_epoch):
                features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.model.features: features_batch}
                gen_cap = sess.run(sampled_labels, feed_dict=feed_dict)
                all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap
                gen_r5_result = sess.run(rank5_results, feed_dict=feed_dict)
                all_rank5_results[i * self.batch_size:(i + 1) * self.batch_size] = gen_r5_result

            test_minus_result = all_gen_cap - labels
            n_labels = len(labels.flatten())
            acc = len(test_minus_result[test_minus_result == 0]) / float(n_labels)
            num_equal_rank5 = np.equal(np.squeeze(all_rank5_results), labels)
            acc_r5 = np.sum(num_equal_rank5) / float(n_labels)
            diff = np.sum(np.abs(test_minus_result)) / float(n_labels)
            print "Acc(rank@1): ", acc
            print "Acc(rank@5): ", acc_r5
            print "Diff: ", diff
            print "Elapsed time: ", time.time() - start_t
            if save_sampled_labels:
                save_pickle(all_gen_cap, "./data/baseline/sample_labels.pkl")

    def inference(self, data, split='train', attention_visualization=True, save_sampled_labels=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - labels: labels of shape (24210, 17)
            - idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_labels: Mapping feature to labels (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_labels: If True, save sampled labels to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample labels
        sampled_labels = self.model.build_sampler(max_len=20)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()
            saver.restore(sess, './model/lstm/model-20')
            features_batch, image_files = sample_coco_minibatch_inference(data, self.batch_size)
            feed_dict = {self.model.features: features_batch}
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_labels], feed_dict)  # (N, max_len, L), (N, max_len)
            decoded = decode_labels(sam_cap, self.model.idx_to_word)
            print "end"
            print decoded