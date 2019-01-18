# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from log_config import logger


# Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
  
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], 3)
    return net


def encode(inputs, is_training=True):

    end_points = {}
  
    with tf.variable_scope('encode'):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                
                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net
        
                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net
                
                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net
                
                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net
                
                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net
                
                # net = block8(net, activation_fn=None)
                # end_points['Mixed_8b'] = net
                #
                # with tf.variable_scope('Logits'):
                #     end_points['PrePool'] = net
                #     #pylint: disable=no-member
                #     net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                #                           scope='AvgPool_1a_8x8')
                #     net = slim.flatten(net)
                #
                #     net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                #                        scope='Dropout')
                #
                #     end_points['PreLogitsFlatten'] = net
                #
                # net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None,
                #         scope='Bottleneck', reuse=False)

    vars = tf.trainable_variables(scope="encode")
    return net, vars


def feature(inputs, scope):
    bottleneck_layer_size = 256
    end_points = {}

    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
            net = slim.conv2d(inputs, 1024, 3, scope='Conv2d_1')
            end_points['Conv2d_1_3x3'] = net

            net = slim.conv2d(net, 512, 3, scope='Conv2d_2')
            end_points['Conv2d_2_3x3'] = net

            net = slim.conv2d(net, 256, 3, scope='Conv2d_3')
            end_points['Conv2d_3_3x3'] = net

            net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_4')
            net = slim.flatten(net)
            net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, scope='Bottleneck_5', reuse=False)

            end_points['PreLogitsFlatten'] = net

    vars = tf.trainable_variables(scope=scope)
    return net, vars

def distil_feature(inputs):
    return feature(inputs, 'distil_feature')

def dispel_feature(inputs):
    return feature(inputs, 'dispel_feature')


def classify(inputs, scope, class_num):
    end_points = {}
    with tf.variable_scope(scope):
        net = slim.fully_connected(inputs, class_num, activation_fn=None)
        end_points['fc'] = net

    vars = tf.trainable_variables(scope=scope)
    return net, vars


def distil_classify(inputs, class_num):
    return classify(inputs, 'distil_classify', class_num)

def dispel_classify(inputs, class_num):
    return classify(inputs, 'dispel_classify', class_num)


def decode(f_t, f_p):
    feature = tf.concat([f_t, f_p], 1)
    with tf.variable_scope('decode'):
        net = tf.reshape(feature, [-1, 1, 1, 512])
        net = slim.layers.conv2d_transpose(net, 512, 4, stride=1, padding='VALID')  # 4*4*512

        # net = slim.conv2d(net, 512, 3, stride=1, scope='Conv2d_1')  # 4*4*512
        net = slim.repeat(net, 2, slim.conv2d, 512, 3, scope='Conv2d_1')
        net = tf.image.resize_images(images=net, size=[8, 8])   # 8*8*512

        net = slim.repeat(net, 3, slim.conv2d, 512, 3, scope='Conv2d_2')  # 8*8*512
        net = tf.image.resize_images(images=net, size=[16, 16])   # 16*16*512

        net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='Conv2d_3')  # 16*16*256
        net = tf.image.resize_images(images=net, size=[32, 32])   # 32*32*256

        net = slim.repeat(net, 3, slim.conv2d, 256, 3, scope='Conv2d_4')   # 32*32*256
        net = tf.image.resize_images(images=net, size=[64, 64])   # 64*64*256

        net = slim.repeat(net, 3, slim.conv2d, 128, 3, scope='Conv2d_5')  # 64*64*128
        net = tf.image.resize_images(images=net, size=[128, 128])   # 128*128*128

        net = slim.repeat(net, 2, slim.conv2d, 64, 3, scope='Conv2d_6')  # 128*128*64
        net = tf.image.resize_images(images=net, size=[256, 256])   # 256*256*64

        net = slim.conv2d(net, 32, 3, stride=1, scope='Conv2d_7')  # 256*256*32
        net = slim.conv2d(net, 3, 1, stride=1, scope='Conv2d_8')    # 256*256*3

    vars = tf.trainable_variables(scope='decode')
    return net, vars