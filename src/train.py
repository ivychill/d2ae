# implementation of D2AE

# import importlib
import argparse
import sys
import tensorflow as tf
from data import Dataset
from graph import Graph
from log_config import logger


gpu_memory_fraction = 1.0
batch_size = 64
epoch_num = 100
image_size = 256


def main(args):

    dataset = Dataset(batch_size)
    class_num = dataset.class_num()
    image_batch, label_batch = dataset.get_train_batch()
    image_batch = tf.reshape(image_batch, [-1, image_size, image_size, 3])

    glaph_ = Graph(batch_size, class_num)
    train_op = glaph_.inference(image_batch, label_batch)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # with sess.as_default():
    # sess.run(iterator.initializer)
    for epoch in range(epoch_num):
        i_batch, l_batch = sess.run([image_batch, label_batch])
        # logger.debug('i_batch: %s, l_batch: %s' % (i_batch, l_batch))
        _, loss_t, loss_p_adv, loss_p, los_re = sess.run(train_op)
        logger.debug('loss_t: %s, loss_p_adv: %s, loss_p: %s, los_re: %s' % (loss_t, loss_p_adv, loss_p, los_re))

    sess.close()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))