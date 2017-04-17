import tensorflow as tf
import os

label_bytes = 1
label_offset = 0
num_classes = 10
image_size = 32
img_depth = 3


def get_random_image_label_from_queue(data_path):
    image_bytes = image_size * image_size * img_depth
    record_bytes = label_bytes + label_offset + image_bytes

    filenames = [os.path.join(data_path, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    file_queue = tf.train.string_input_producer(filenames, shuffle=True)

    # Read images from files in the filename queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # Convert images to dense labels and processed images.
    record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
    label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)

    # Convert from string to [img_depth * height * width] to [img_depth, height, width]
    depth_major = tf.reshape(tf.slice(record, [label_bytes], [image_bytes]), [img_depth, image_size, image_size])
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, label


def build_distorted_input(data_path, batch_size, mode):
    image, label = get_random_image_label_from_queue(data_path)

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size + 4, image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, 3])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)

        images_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, img_depth], [1]]
            )
        num_threads = 16
    elif mode == 'eval':
        image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
        image = tf.image.per_image_standardization(image)
        images_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, img_depth], [1]])
        num_threads = 1

    images_enqueue_op = images_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(images_queue, [images_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = images_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(tf.concat(values=[indices, labels], axis=1), [batch_size, num_classes], 1.0, 0.0)

    return images, labels


def build_inputs(data_path, batch_size):

    image, label = get_random_image_label_from_queue(data_path)

    images_queue = tf.RandomShuffleQueue(
        capacity=3 * batch_size,
        min_after_dequeue=1,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_size, image_size, img_depth], [1]])
    num_threads = 1

    images_enqueue_op = images_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(images_queue, [images_enqueue_op] * num_threads))

    # Read 'batch' labels + images from the example queue.
    images, labels = images_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    labels = tf.sparse_to_dense(tf.concat(values=[indices, labels], axis=1), [batch_size, num_classes], 1.0, 0.0)

    return images, labels
