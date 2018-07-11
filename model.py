import tensorlayer as tl
import tensorflow as tf
import time
from tensorlayer.layers import *


def VGG19(rgb, reuse):
    # rgb: image inputs pixel value range [0, 1]
    VGG_MEAN = [103.939, 116.779, 123.68]

    with tf.variable_scope('VGG19', reuse=reuse):
        start_time = time.time()
        print ("Start to build model...")
        rgb_scaled = rgb *255.0

        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
                ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                    ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = InputLayer(bgr, name='input')
        """ conv1 """
        network = Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        """ conv5 """
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = FlattenLayer(network, name='flatten')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        conv = network
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv

def generator(feature_map, is_train=False, reuse=False):

    w_init = tf.truncated_normal_initializer(stddev=0.02)
    g_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)

    image_size = 224

    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16),

    gf_dim = 32

    c_dim = 3

    assert feature_map.outputs.get_shape().as_list()[1:] == [1000]

    # make sure the size matches if the size of current batch is not batch size
    batch_size = feature_map.outputs.get_shape().as_list()[0]

    filter_size = (5, 5)
    strides = (2, 2)

    with tf.variable_scope('generator', reuse=reuse):

        # (1000,)
        net_in = InputLayer(feature_map.outputs, name='g/in')

        # (14*14*32*15=14*14*512, )
        net_h0 = DenseLayer(net_in, n_units=gf_dim*16*s16*s16, W_init=w_init,
                            act = tf.identity, name='g/h0/lin')


        # (14, 14, 512)
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*16], name="g/h0/reshape")

        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h0/batch_norm')

        # (28, 28, 256)
        net_h1 = DeConv2d(net_h0, n_filter=gf_dim*8, filter_size=filter_size, out_size=(s8, s8), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h1/deconv2d')

        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h1/batch_norm')

        # (56, 56, 128)
        net_h2 = DeConv2d(net_h1, n_filter=gf_dim*4, filter_size=filter_size, out_size=(s4, s4), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h2/deconv2d')

        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h2/batch_norm')

        # (112, 112, 64)
        net_h3 = DeConv2d(net_h2, n_filter=gf_dim*2, filter_size=filter_size, out_size=(s2, s2), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h3/deconv2d')

        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h3/batch_norm')

        # (224, 224, 3)
        net_h4 = DeConv2d(net_h3, n_filter=c_dim, filter_size=filter_size, out_size=(image_size, image_size), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h4/deconv2d')

        logits = net_h4.outputs
        net_h4.outputs = tf.nn.tanh(net_h4.outputs)

        return net_h4, logits

def discriminator(inputs, is_train=True, reuse=False):

    df_dim = 64

    w_init = tf.truncated_normal_initializer(stddev=0.02)

    gamma_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)

    filter_size = (5, 5)

    strides = (2, 2)

    with tf.variable_scope('discriminator', reuse=reuse):

        net_in = InputLayer(inputs=inputs, name='d/in')

        # (112, 112, 64)
        net_h0 = Conv2d(net_in, n_filter=df_dim, filter_size=filter_size, strides=strides,
                        act = lambda x: tl.act.lrelu(x, 0.2), padding='SAME', W_init=w_init, name='d/h0/conv2d')


        # (56, 56, 128)
        net_h1 = Conv2d(net_h0, n_filter=df_dim*2, filter_size=filter_size, strides=strides,
                        act=None, padding='SAME', W_init=w_init, name='d/h1/conv2d')

        net_h1 = BatchNormLayer(net_h1, act=lambda x:tl.act.lrelu(x, 0.2), is_train=is_train,
                                gamma_init=gamma_init, name='d/h1/batch_norm')

        # (28, 28, 256)
        net_h2 = Conv2d(net_h1, n_filter=df_dim*4, filter_size=filter_size, strides=strides,
                        act=None, padding='SAME', W_init=w_init, name='d/h2/conv2')

        net_h2 = BatchNormLayer(net_h2, act=lambda x:tl.act.lrelu(x, 0.2), is_train=is_train,
                                gamma_init=gamma_init, name='d/h2/batch_norm')

        # (14, 14, 512)
        net_h3 = Conv2d(net_h2, n_filter=df_dim*8, filter_size=filter_size, strides=strides,
                        act=None, padding='SAME', W_init=w_init, name='d/h3/conv2')

        net_h3 = BatchNormLayer(net_h3, act=lambda x:tl.act.lrelu(x, 0.2), is_train=is_train,
                                gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity, W_init=w_init, name='d/h4/lin_sigmoid')

        logits = net_h4.outputs

        net_h4.outputs = tf.nn.sigmoid(net_h4.outputs)

    return net_h4, logits







