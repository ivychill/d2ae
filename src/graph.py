# implementation of D2AE

import tensorflow as tf
import numpy as np
import network
from data import Dataset

keep_probability = 0.8

class Graph():
    def __init__(self, batch_size, class_num):
        self.lambda_t = 1.0
        self.lambda_p = 0.1
        self.lambda_x = 1.81e-5
        self.learning_rate = 1e-4
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = batch_size
        # self.dataset = Dataset(self.batch_size)
        self.class_num = class_num  # 2213

    # def setup(self):
    #     image_batch, label_batch, iterator = self.dataset.get_train_batch()

    def inference(self, image_batch, label_batch):
        embedding, vars_encode = network.encode(image_batch)

        # identity distilling Li
        feature_distil, vars_distil_feature = network.distil_feature(embedding)
        logits_distil, vars_distil_classify = network.distil_classify(feature_distil, self.class_num)
        cross_entropy_distil = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits_distil, name='cross_entropy_per_example')
        distil_loss = tf.reduce_mean(cross_entropy_distil, name='cross_entropy')
        distil_loss = distil_loss * self.lambda_t

        # identity dispelling Li_idv
        feature_dispel, vars_dispel_feature = network.dispel_feature(embedding)
        logits_dispel, vars_dispel_classify = network.dispel_classify(feature_dispel, self.class_num)
        cross_entropy_dispel_adv = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits_dispel, name='cross_entropy_per_example')
        dispel_adv_loss = tf.reduce_mean(cross_entropy_dispel_adv, name='cross_entropy')
        dispel_adv_loss = dispel_adv_loss * self.lambda_p

        label_uniform = np.full((self.batch_size, self.class_num), 1.0/self.class_num)
        cross_entropy_dispel = tf.nn.softmax_cross_entropy_with_logits(
            labels=label_uniform, logits=logits_dispel, name='cross_entropy_per_example')
        dispel_loss = tf.reduce_mean(cross_entropy_dispel, name='cross_entropy')
        dispel_loss = dispel_loss * self.lambda_p

        # reconstruction
        outputs, vars_decode = network.decode(feature_distil, feature_dispel)
        reconstruction_loss = 1/2 * tf.losses.mean_squared_error(image_batch, outputs)
        reconstruction_loss = reconstruction_loss * self.lambda_x

        # dict(dict1, **dict2): merge collections
        # vars_distil = dict(vars_distil_feature, **vars_distil_classify)
        # vars_distil_loss = dict(vars_encode, **vars_distil)
        # vars_dispel_adv_loss = vars_dispel_feature
        # vars_dispel_loss = dict(vars_encode, **vars_dispel_classify)
        # vars_reconstruction = dict(dict(vars_encode, **vars_distil_feature), **dict(vars_dispel_feature, **vars_decode))
        vars_distil_loss = vars_encode + vars_distil_feature + vars_distil_classify
        vars_dispel_adv_loss = vars_dispel_classify
        vars_dispel_loss = vars_encode + vars_dispel_feature
        vars_reconstruction_loss = vars_encode + vars_distil_feature + vars_dispel_feature + vars_decode

        opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        grad_distil = opt.compute_gradients(distil_loss, var_list=vars_distil_loss)
        grad_dispel_adv = opt.compute_gradients(dispel_adv_loss, var_list=vars_dispel_adv_loss)
        grad_dispel = opt.compute_gradients(dispel_loss, var_list=vars_dispel_loss)
        grad_reconstruction = opt.compute_gradients(reconstruction_loss, var_list=vars_reconstruction_loss)

        grad = grad_distil + grad_dispel_adv + grad_dispel + grad_reconstruction
        apply_gradient_op = opt.apply_gradients(grad, global_step=self.global_step)

        return [apply_gradient_op, distil_loss, dispel_adv_loss, dispel_loss, reconstruction_loss]