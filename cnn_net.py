from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

HyperParameters = namedtuple('HPS',
                             'batch_size, num_classes, weight_decay_rate, optimizer, learning_rate, min_learning_rate')


class CNN_NET(object):
    """ Build Net model """

    def __init__(self, hps, images, labels, mode, arch):
        """
                Net Constructor

                Args:
                        hps: Hyperparameters
                        images: Batches of images => [batch_size, img_size, img_size, 3]
                        labels: Batches of labels => [batch_size, num_classes]
                        mode: 'train' or 'eval'
        """
        self.hps = hps
        self.images = images
        self.labels = labels
        self.mode = mode
        self.arch = arch
        self.conv_list = []

        self._extra_train_ops = []

    def build_graph(self):
        # Returns and create (if necessary) the global step variable
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        
        if self.arch == 1:
            self._build_model_1()
        elif self.arch == 2:
            self._build_model_2()
        elif self.arch == 3:
            self._build_model_3()
        else:
            raise("The architecture doesn't exist!")

        if self.mode == 'train':
            self._build_train_op()

        self.summaries = tf.summary.merge_all()

    def _build_model_1(self):
        """ Build the core model within the graph """

        # Conv 1
        self.conv1 = self._conv_layer('conv1', self.images, filter_size=3, in_filters=3, out_filters=64, strides=[1, 1, 1, 1])

        # Pool 1
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # Norm 1
        self.norm1 = self._local_response_norm('norm1', self.pool1)

        # Conv 2
        self.conv2 = self._conv_layer('conv2', self.norm1, 3, 64, 64, [1, 1, 1, 1])

        # Norm 2
        self.norm2 = self._local_response_norm('norm2', self.conv2)

        # Pool 2
        self.pool2 = tf.nn.max_pool(self.norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # Conv 3
        self.conv3 = self._conv_layer('conv3', self.pool2, 3, 64, 64, [1, 1, 1, 1])

        # Pool 3
        self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # Norm 3
        self.norm3 = self._local_response_norm('norm3', self.pool3)

        # FC 1
        self.fc1 = self._fully_connected_layer('fc1', self.norm3, 384)

        # Leaky Relu
        self.fc1 = self._relu(self.fc1)

        # FC 2
        self.fc2 = self._fully_connected_layer('fc2', self.fc1, self.hps.num_classes)
        self.predictions = tf.nn.softmax(logits=self.fc2)

        self.cost = self._loss()


    def _build_model_2(self):
        """ Build the core model within the graph """

        # Conv 1
        self.conv1 = self._conv_layer('conv1', self.images, filter_size=3, in_filters=3, out_filters=64, strides=[1, 1, 1, 1])

        # Pool 1
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # Conv 2
        self.conv2 = self._conv_layer('conv2', self.pool1, 3, 64, 64, [1, 1, 1, 1])

        # Pool 2
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # Conv 3
        self.conv3 = self._conv_layer('conv3', self.pool2, 3, 64, 64, [1, 1, 1, 1])

        # Pool 3
        self.pool3 = tf.nn.max_pool(self.conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # FC 1
        self.fc1 = self._fully_connected_layer('fc1', self.pool3, 384)

        # Leaky Relu
        self.fc1 = self._relu(self.fc1)

        # FC 2
        self.fc2 = self._fully_connected_layer('fc2', self.fc1, self.hps.num_classes)
        self.predictions = tf.nn.softmax(logits=self.fc2)

        self.cost = self._loss()
    

    def _build_model_3(self):
        """ Build the core model within the graph """

        # Conv 1
        self.conv1 = self._conv_layer('conv1', self.images, filter_size=5, in_filters=3, out_filters=16, strides=[1, 3, 3, 1])

        # Pool 1
        self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        # Conv 2
        self.conv2 = self._conv_layer('conv2', self.pool1, 3, 16, 16, [1, 1, 1, 1])

        # Pool 2
        self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # FC 1
        self.fc1 = self._fully_connected_layer('fc1', self.pool2, 384)

        # Leaky Relu
        self.fc1 = self._relu(self.fc1)

        # FC 2
        self.fc2 = self._fully_connected_layer('fc2', self.fc1, self.hps.num_classes)
        self.predictions = tf.nn.softmax(logits=self.fc2)

        self.cost = self._loss()


    def _loss(self):
        with tf.variable_scope('costs'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.fc2,
                                                                    labels=self.labels)
            cost = tf.reduce_mean(cross_entropy, name='corss_entropy')
            cost += self._l2_decay()

            tf.summary.scalar('cost', cost)

            return cost


    def _build_train_op(self):
        """ Build training specific ops for the graph """

        # [?]
        self.learning_rate = tf.constant(self.hps.learning_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.learning_rate)

        grads = tf.gradients(self.cost, tf.trainable_variables())

        if self.hps.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.hps.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(
                self.learning_rate, momentum=0.9)

        apply_grad_op = optimizer.apply_gradients(zip(grads, tf.trainable_variables()),
                                                  global_step=self.global_step,
                                                  name='train_step')

        train_ops = [apply_grad_op] + self._extra_train_ops
        self.train_op = tf.group(*train_ops)


    def _conv_layer(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name) as scope:
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(name='weights',
                                     shape=[filter_size, filter_size,
                                            in_filters, out_filters],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(
                                         stddev=np.sqrt(2.0 / n))
                                     )

            conv = tf.nn.conv2d(input=x, filter=kernel,
                                strides=strides, padding='SAME')
            biases = tf.get_variable(name='biases',
                                     shape=[out_filters],
                                     initializer=tf.constant_initializer(0.0))

            pre_activation = tf.nn.bias_add(conv, biases)
            conv = tf.nn.relu(pre_activation, name=scope.name)

            self.conv_list.append(conv)
            
            return conv


    def _local_response_norm(self, name, x):
        return tf.nn.lrn(x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


    def _max_pool(self, name, bottom):
        return


    def _avg_pool(self, name, bottom):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    def _fully_connected_layer(self, name, x, out_dim):
        with tf.variable_scope(name) as scope:
            x = tf.reshape(x, [self.hps.batch_size, -1])
            w = tf.get_variable('weights',
                                [x.get_shape()[1], out_dim],
                                dtype=tf.float32,
                                initializer=tf.uniform_unit_scaling_initializer(
                                    factor=1.0)
                                )
            b = tf.get_variable(
                'biases', [out_dim], initializer=tf.constant_initializer())

            return tf.nn.xw_plus_b(x, w, b)


    def _relu(self, x, leakiness=0.0):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leak_relu')


    def _l2_decay(self):
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'weights') > 0:
                costs.append(tf.nn.l2_loss(var))

        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))


    def _batch_norm(self, name, x):
        """ TEST """
        with tf.variable_scope(name):
            params_shape = [x.get_shape()[-1]]

            beta = tf.get_variable('beta', params_shape, tf.float32,
                                   initializer=tf.constant_initializer(0.0, tf.float32))
            gamma = tf.get_variable('gamma', params_shape, tf.float32,
                                    initializer=tf.constant_initializer(1.0, tf.float32))

            if self.mode == 'train':
                mean, variance = tf.nn.moments(
                    x=x, axes=[0, 1, 2], name='moments')

                moving_mean = tf.get_variable('moving_mean', params_shape, tf.float32,
                                              initializer=tf.constant_initializer(0.0, tf.float32), trainable=False)
                moving_variance = tf.get_variable(
                    'moving_variance', params_shape, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)

                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    variable=moving_mean, value=mean, decay=0.9))
                self._extra_train_ops.append(moving_averages.assign_moving_average(
                    variable=moving_variance, value=variance, decay=0.9))
            else:
                mean = tf.get_variable('moving_mean', params_shape, tf.float32, initializer=tf.constant_initializer(
                    0.0, tf.float32), trainable=False)
                variance = tf.get_variable('moving_variance', params_shape, tf.float32,
                                           initializer=tf.constant_initializer(1.0, tf.float32), trainable=False)
                tf.summary.histogram(mean.op.name, mean)
                tf.summary.histogram(mean.op.name, variance)

            y = tf.nn.batch_normalization(
                x, mean, variance, beta, gamma, 0.001)
            y.set_shape(x.get_shape())

            return y

