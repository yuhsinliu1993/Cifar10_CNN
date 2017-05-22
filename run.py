import cnn_net
import cifar10_input
import tensorflow as tf
import time
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

data_path = 'data/cifar-10-batches-bin'


def train(hps, mode, arch):
    images, labels = cifar10_input.build_distorted_input(data_path, hps.batch_size, mode)

    # Build the model and training ops
    model = cnn_net.CNN_NET(hps, images, labels, mode, arch)
    model.build_graph()

    y_true = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))

    summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                             output_dir=train_dir,
                                             summary_op=tf.summary.merge([model.summaries,
                                                                          tf.summary.scalar('Accuracy', accuracy)]))

    logging_hook = tf.train.LoggingTensorHook(tensors={
        'step': model.global_step,
        'loss': model.cost,
        'accuracy': accuracy},
        every_n_iter=100
    )

    class _LearningRateSetterHook(tf.train.SessionRunHook):
        """ Decay the learning rate after certain amount of training steps """

        def begin(self):
            self._learning_rate = 0.1

        def before_run(self, run_context):
            # Because we will change the learning rate, we need to
            # feed it to the model while running the session
            return tf.train.SessionRunArgs(model.global_step,
                                           feed_dict={model.learning_rate: self._learning_rate}
                                           )

        def after_run(self, run_context, run_values):
            train_step = run_values.results  # trian_step == global_step

            if train_step < 20000:
                self._learning_rate = 0.1
            elif train_step < 40000:
                self._learning_rate = 0.01
            elif train_step < 60000:
                self._learning_rate = 0.001
            else:
                self._learning_rate = 0.0001

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir,
        hooks=[logging_hook, _LearningRateSetterHook()],
        chief_only_hooks=[summary_hook],
        save_summaries_steps=0  # To disable the default SummarySaverHook
    ) as mon_sess:
        while not mon_sess.should_stop():
            mon_sess.run(model.train_op)


def evaluate(hps, mode, eval_once, arch):
    images, labels = cifar10_input.build_distorted_input(
        data_path, hps.batch_size, mode)

    model = cnn_net.CNN_NET(hps, images, labels, mode, arch)
    model.build_graph()

    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(eval_dir)

    sess = tf.Session()
    # Start all the queue runners collected in the graph
    tf.train.start_queue_runners(sess)

    best_accuracy = 0.0
    while True:
        try:
            ckpt_state = tf.train.get_checkpoint_state(
                checkpoint_dir=checkpoint_dir)
        except tf.errors.OutOfRangeError as e:
            tf.logging.erorr('Cat\'t restore checkpoint: %s', e)
            continue

        if not (ckpt_state and ckpt_state.model_checkpoint_path):
            tf.logging.info('No model to eval at %s', checkpoint_dir)
            continue

        tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)

        saver.restore(sess, ckpt_state.model_checkpoint_path)

        total_prediction, correct_predicion = 0.0, 0.0
        for _ in range(50):
            (summaries, loss, predictions, y_true, train_step) = sess.run(
                [model.summaries, model.cost, model.predictions, model.labels, model.global_step])

            y_true = np.argmax(y_true, axis=1)
            predictions = np.argmax(predictions, axis=1)
            correct_predicion += np.sum(y_true == predictions)
            total_prediction += predictions.shape[0]

        accuracy = 1.0 * correct_predicion / total_prediction
        best_accuracy = max(accuracy, best_accuracy)

        accuracy_summary = tf.Summary()
        best_accuracy_summary = tf.Summary()

        accuracy_summary.value.add(tag='Accuracy', simple_value=accuracy)
        summary_writer.add_summary(accuracy_summary, train_step)
        best_accuracy_summary.value.add(
            tag='Best Accuracy', simple_value=best_accuracy)
        summary_writer.add_summary(best_accuracy_summary, train_step)

        summary_writer.add_summary(summaries, train_step)

        tf.logging.info('loss: %.3f, accuracy: %.3f, best accuracy: %.3f' % (loss, accuracy, best_accuracy))
        summary_writer.flush()

        if eval_once:
            break

        time.sleep(60)


def _plot(conv, n=0):
    num_filters = conv.shape[3]
    num_grids = int(math.ceil(math.sqrt(num_filters)))

    fig, axes = plt.subplots(num_grids, num_grids, figsize=(20, 20))

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = conv[n, :, :, i]
            ax.imshow(img, interpolation='nearest')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def _get_conv_feature_maps(hps, arch):
    image, label = cifar10_input.build_inputs(data_path, hps.batch_size)

    model = cnn_net.CNN_NET(hps, image, label, 'eval', arch)
    model.build_graph()

    sess = tf.Session()

    tf.train.start_queue_runners(sess)

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    conv_maps = sess.run(model.conv_list)
    images = sess.run([model.images])

    return conv_maps, images


def plot_feature_maps(hps, arch):
    conv_maps, images = _get_conv_feature_maps(hps, arch)

    for i in range(len(conv_maps)):
        _plot(conv_maps[i], 0)


def _get_kwargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="Specify the mode. 'train' or 'eval' or 'plot'", required=True)
    parser.add_argument("-e", "--evalonce", action='store_true', help="Specify to only do one evaluation")
    parser.add_argument("-a", "--architecture", type=int, help="Specify which model you want to use, default is model 1", default=1)
    return vars(parser.parse_args())


def run(**kwargs):
    global train_dir
    global eval_dir
    global checkpoint_dir

    if not kwargs:
        kwargs = _get_kwargs()

    hps = cnn_net.HyperParameters(batch_size=128,
                                  num_classes=10,
                                  learning_rate=0.1,
                                  min_learning_rate=0.0001,
                                  weight_decay_rate=0.0002,
                                  optimizer='momentum'
                                  )

    arch = kwargs['architecture']

    train_dir = 'models/model%d/train' % arch
    eval_dir = 'models/model%d/eval' % arch
    checkpoint_dir = 'models/model%d/checkpoints' % arch

    eval_once = False
    if kwargs['evalonce']:
        eval_once = True

    if kwargs['mode'] == 'train':
        train(hps, kwargs['mode'], arch)
    elif kwargs['mode'] == 'eval':
        evaluate(hps, kwargs['mode'], eval_once, arch)
    elif kwargs['mode'] == 'plot':
        plot_feature_maps(hps, arch)

run()
