import facenet
import tensorflow as tf
# from log_config import logger


class Dataset():
    def __init__(self, batch_size):
        data_dir = '~/datasets/casia/casia_maxpy_mtcnnpy_256'
        self.batch_size = batch_size
        self.train_set = facenet.get_dataset(data_dir)

    def class_num(self):
        return len(self.train_set)

    def _parse_function(self, image_path, label):
        image_string = tf.read_file(image_path)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image = tf.to_float(image_decoded)
        # image_decoded = tf.cond(
        #     tf.image.is_jpeg(image_string),
        #     lambda: tf.image.decode_jpeg(image_string, channels=3),
        #     lambda: tf.image.decode_png(image_string, channels=3))
        # image_resized = tf.image.resize_images(image_decoded, [90, 90])
        # return image_resized
        return image, label

    def get_train_batch(self):
        # filenames = ["/var/data/image1.jpg", "/var/data/image2.jpg", ...]
        # labels = [0, 37, 29, 1, ...]
        image_path_list, label_list = facenet.get_image_paths_and_labels(self.train_set)
        dataset = tf.data.Dataset.from_tensor_slices((image_path_list, label_list))
        dataset = dataset.map(self._parse_function)
        dataset = dataset.batch(self.batch_size)
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        return image_batch, label_batch