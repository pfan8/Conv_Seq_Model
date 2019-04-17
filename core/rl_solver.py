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
from bleu import evaluate,evaluate_labels


class CaptioningSolver(object):
    def __init__(self, model, data, val_data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - labels: labels of shape (400000, 17)
                - image_idxs: Indices for mapping caption to image of shape (400000, )
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

        # build graphs for training model and sampling labels
        loss = self.model.build_model()
        tf.get_variable_scope().reuse_variables()
        alphas, betas, sampled_labels, loss = self.model.build_multinomial_sampler()

        _, _, greedy_caption = self.model.build_sampler(max_len=7)

        rewards = tf.placeholder(tf.float32, [None])
        base_line = tf.placeholder(tf.float32, [None])

        # train op
        with tf.name_scope('optimizer'):


            optimizer = self.optimizer(learning_rate=self.learning_rate)
            norm = tf.shape(rewards)[0] * 7
            sum_loss = tf.abs(tf.reduce_sum(tf.mul(tf.transpose(loss, [2, 1, 0]),( rewards - base_line))) / tf.cast(norm, dtype=tf.float32))

            grads_rl = tf.gradients(sum_loss,tf.trainable_variables(),aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            grads_and_vars = list(zip(grads_rl,tf.trainable_variables()))

            train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        # metrics
        greedy_caption = tf.cast(greedy_caption, dtype=tf.int32)
        metric_add_result = self.model.labels + greedy_caption
        equal_to_0 = tf.equal(metric_add_result, 0)
        equal_to_2 = tf.equal(metric_add_result, 2)
        true_negative = tf.reduce_sum(tf.cast(equal_to_0, tf.int32))
        true_positive = tf.reduce_sum(tf.cast(equal_to_2, tf.int32))
        metric_minus_result = self.model.labels - greedy_caption
        equal_to_1 = tf.equal(metric_minus_result, 1)
        equal_to_n1 = tf.equal(metric_minus_result, -1)
        false_negative = tf.reduce_sum(tf.cast(equal_to_1, tf.int32))
        false_positive = tf.reduce_sum(tf.cast(equal_to_n1, tf.int32))

        # summary op


        print "The number of epoch: %d" % self.n_epochs
        print "Data size: %d" % n_examples
        print "Batch size: %d" % self.batch_size
        print "Iterations per epoch: %d" % n_iters_per_epoch

        config = tf.ConfigProto(allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction=0.9
        config.gpu_options.allow_growth = True
        config.gpu_options.allocator_type = 'BFC'
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            if self.pretrained_model is not None:
                print "Start training with pretrained Model.."
                saver.restore(sess, self.pretrained_model)

            prev_loss = -1
            curr_loss = 0
            prev_acc = -1
            curr_acc = 0
            max_F1 = 0.0
            start_t = time.time()
            max_len = 7

            for e in range(self.n_epochs):
                rand_idxs = np.random.permutation(n_examples)
                labels = np.array(labels[rand_idxs])
                features = np.array(features[rand_idxs])
                b_for_eval = []

                TP,FP,TN,FN = 0,0,0,0

                for i in range(n_iters_per_epoch):
                    labels_batch = np.array(labels[i * self.batch_size:(i + 1) * self.batch_size])
                    features_batch = np.array(features[i * self.batch_size:(i + 1) * self.batch_size])
                    ground_truths = [labels[j] for j in range(len(features_batch))]
                    ref_decoded = ground_truths

                    feed_dict = {self.model.features: features_batch, self.model.labels: labels_batch,
                                self.model.sample_labels: labels_batch}

                    samples, greedy_words = sess.run([sampled_labels, greedy_caption],
                                                             feed_dict)
                    all_decoded, greedy_decoded = samples, greedy_words
                    # write summary for tensorboard visualization

                    r = [evaluate_labels(k, v)  for k, v in zip(ref_decoded, all_decoded)]
                    b = [evaluate_labels(k, v) for k, v in zip(ref_decoded, greedy_decoded)]

                    b_for_eval.extend(b)
                    feed_dict = {rewards: r, base_line: b,
                                 self.model.features: features_batch, self.model.labels: labels_batch}  # write summary for tensorboard visualization
                    _,l = sess.run([train_op, sum_loss], feed_dict)
                    tp,fp,tn,fn = sess.run([true_positive, false_positive,\
                                            true_negative, false_negative], feed_dict)
                    
                    curr_loss += l
                    TP += tp
                    FP += fp
                    TN += tn
                    FN += fn
                print str(np.mean(np.array(b_for_eval)))
                # print out BLEU scores and file write
                print "================epoch %d===============" % e
                print "Elapsed time: ", time.time() - start_t
                # if self.print_bleu:
                #     print "b" + str(np.mean(np.array(b)))
                #     print "r" + str(np.mean(np.array(r)))
                #     all_gen_cap = np.ndarray((val_features.shape[0], 128))
                #     for k in range(n_iters_val):
                #         features_batch = val_features[k * self.batch_size:(k + 1) * self.batch_size]
                #         labels_words_batch = np.array(val_labels[k * self.batch_size:(k + 1) * self.batch_size])

                #         feed_dict = {self.model.features: features_batch, self.model.labels: labels_words_batch}
                #         gen_cap = sess.run(greedy_caption, feed_dict=feed_dict)
                #         all_gen_cap[k * self.batch_size:(k + 1) * self.batch_size] = gen_cap
                #     masks,all_decoded = decode_labels_for_blue(all_gen_cap, self.model.idx_to_word)
                #     for s in range(5):
                #         print all_decoded[-s-1]
                #     save_pickle(all_decoded, "./data/val/val.candidate.labels.pkl")
                #     scores = evaluate(data_path='./data', split='val', get_scores=True)
                #     write_bleu(scores=scores, path=self.model_path, epoch=e)

                # summary metrics
                print "Previous epoch loss: ", prev_loss
                print "Current epoch loss: ", curr_loss
                assert (TP+FP+TN+FN) == n_examples * max_len
                curr_acc = (TP + TN) / float(n_examples * max_len)
                print "Previous epoch acc: ", prev_acc
                print "Current epoch acc: ", curr_acc
                print "True Positive: ", TP
                print "False Positive: ", FP
                print "True Negative: ", TN
                print "False Negative: ", FN
                recall = TP / float(TP + FN )
                print "Recall:", recall
                F1 = curr_acc * recall * 2 / (curr_acc + recall)
                print "F1:", F1
                print "Elapsed time: ", time.time() - start_t
                prev_loss = curr_loss
                prev_acc = curr_acc
                curr_loss = 0

                # save model's parameters
                if max_F1 < F1:
                    max_F1= F1
                    model_name = "model_" + "{:.2f}".format(max_F1) + '_F1'
                    saver.save(sess, os.path.join(self.model_path, model_name), global_step=e + 1)
                    print "model of %.2f F1 saved." % (max_F1)

    def test(self, data, split='train', attention_visualization=True, save_sampled_labels=False):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - labels: labels of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
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
        alphas, betas, sampled_labels = self.model.build_sampler(max_len=7)  # (N, max_len, L), (N, max_len)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            tf.initialize_all_variables().run()
            saver = tf.train.Saver()
            saver.restore(sess, self.test_model)
            
            # features_batch = features[batch_size]
            # feed_dict = {self.model.features: features}
            # alps, bts, sam_cap = sess.run([alphas, betas, sampled_labels], feed_dict)  # (N, max_len, L), (N, max_len)
            # decoded = decode_labels(sam_cap, self.model.idx_to_word)

            all_gen_cap = np.ndarray((labels.shape[0], 7))
            start_t = time.time()
            for i in range(n_iters_per_epoch):
                features_batch = features[i * self.batch_size:(i + 1) * self.batch_size]
                feed_dict = {self.model.features: features_batch}
                gen_cap = sess.run(sampled_labels, feed_dict=feed_dict)
                all_gen_cap[i * self.batch_size:(i + 1) * self.batch_size] = gen_cap

            test_add_result = all_gen_cap + labels
            TP = len(test_add_result[test_add_result == 2])
            TN = len(test_add_result[test_add_result == 0])
            test_minus_result = all_gen_cap - labels
            FP = len(test_minus_result[test_minus_result == 1])
            FN = len(test_minus_result[test_minus_result == -1])
            n_labels = len(labels.flatten())
            assert (TP+FP+TN+FN) == n_labels
            acc = (TP + TN) / float(n_labels)
            print "Acc: ", acc
            print "True Positive: ", TP
            print "False Positive: ", FP
            print "True Negative: ", TN
            print "False Negative: ", FN
            recall = TP / float(TP + FN)
            print "Recall:", recall
            F1 = acc * recall * 2 / (acc + recall)
            print "F1:", F1
            print "Elapsed time: ", time.time() - start_t
            if save_sampled_labels:
                save_pickle(all_gen_cap, "./data/baseline/rl_sample_labels.pkl")

    def inference(self, data, split='train', attention_visualization=True, save_sampled_labels=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - labels: labels of shape (24210, 17)
            - image_idxs: Indices for mapping caption to image of shape (24210, )
            - features_to_labels: Mapping feature to labels (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_labels: If True, save sampled labels to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample labels
        alphas, betas, sampled_labels = self.model.build_sampler(max_len=20)  # (N, max_len, L), (N, max_len)

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