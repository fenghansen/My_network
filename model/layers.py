import keras
import numpy as np
import keras.backend as K
from keras.layers import Conv2D, Conv2DTranspose, Lambda, UpSampling2D, concatenate, BatchNormalization
from keras.layers import Add, MaxPooling2D, AveragePooling2D, PReLU, LeakyReLU, ReLU, Concatenate
import tensorflow as tf


class SpectralNormalization:
    """
        层的一个包装，用来加上SN。
        引用自苏剑林博客https://kexue.fm/archives/6051#Keras%E5%AE%9E%E7%8E%B0
        SN层的理论详情可见论文https://arxiv.org/abs/1705.10941，
        《Spectral Norm Regularization for Improving the Generalizability of Deep Learning》
        Keras应用：https://github.com/bojone/gan/blob/master/keras/wgan_sn_celeba.py
    """

    def __init__(self, layer):
        self.layer = layer

    def spectral_norm(self, w, r=5):
        w_shape = K.int_shape(w)
        in_dim = np.prod(w_shape[:-1]).astype(int)
        out_dim = w_shape[-1]
        w = K.reshape(w, (in_dim, out_dim))
        u = K.ones((1, in_dim))
        for _ in range(r):
            v = K.l2_normalize(K.dot(u, w))
            u = K.l2_normalize(K.dot(v, K.transpose(w)))
        return K.sum(K.dot(K.dot(u, w), K.transpose(v)))

    def spectral_normalization(self, w):
        return w / self.spectral_norm(w)

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        if not hasattr(self.layer, 'spectral_normalization'):
            if hasattr(self.layer, 'kernel'):
                self.layer.kernel = self.spectral_normalization(self.layer.kernel)
            if hasattr(self.layer, 'gamma'):
                self.layer.gamma = self.spectral_normalization(self.layer.gamma)
            self.layer.spectral_normalization = True
        return self.layer(inputs)


def SubpixelConv2D(name, scale=2):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    :param scale: upsampling scale compared to input_shape. Default=2
    :return:
    """

    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                None if input_shape[1] is None else input_shape[1] * scale,
                None if input_shape[2] is None else input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.depth_to_space(x, scale)

    return Lambda(subpixel, output_shape=subpixel_shape, name=name)


def Subpixel(x, number, scale=2, sn=False):
    """
    使用亚像素卷积进行上采样的Keras组合拳
    :param x: Your Keras Layer input.
    :param number: This number is used to name the Subpixel Layers.
    :param scale: upsampling scale compared to input_shape. Default=2
    :param sn: If you want to use SpectralNormalization
    """
    if sn:
        x = SpectralNormalization(Conv2D(256, kernel_size=3, strides=1, 
            padding='same', name='upSampleConv2d_' + str(number)))(x)
    else:
        x = Conv2D(256, kernel_size=3, strides=1, padding='same', 
                        name='upSampleConv2d_' + str(number))(x)
    x = SubpixelConv2D('upSampleSubPixel_' + str(number), scale)(x)
    x = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(number))(x)
    return x


def upsample(x, number=0, mode='subpixel', scale=2, sn=False):
    """
    使用亚像素卷积进行上采样的Keras组合拳
    :param x: Your Keras Layer input.
    :param number: This number is used to name the Subpixel Layers.
    :param mode: Three choices 'subpixel', 'deconv', 'upsample'
    :param scale: upsampling scale compared to input_shape. Default=2
    :param sn: If you want to use SpectralNormalization
    """
    input_shape = K.int_shape(x)
    n_filters = input_shape[-1]
    if mode == 'subpixel':
        return Subpixel(x, number, scale=scale, sn=sn)
    elif mode == 'deconv':
        if sn:
            return SpectralNormalization(Conv2DTranspose(filters=n_filters, 
                                         kernel_size=4, strides=2, padding='same'))(x)
        else:
            return Conv2DTranspose(filters=n_filters, kernel_size=4, strides=2, padding='same')(x)
    else:
        return UpSampling2D()(x)


def downsample(x, n_filters=256, mode='strides', sn=False):
    """
    修改说明：
    2019.10.13: 1.添加了三种下采样方式，包括两种池化(max,average)和最临近(nearest)插值缩小
                2.将名称改为downsample，防止和Keras官方版本重复
    """
    input_shape = K.int_shape(x)
    n_filters = input_shape[-1]
    if mode == 'strides':
        if sn:
            return SpectralNormalization(Conv2D(filters=n_filters, 
                                         kernel_size=3, strides=2, padding='same'))(x)
        else:
            return Conv2D(filters=n_filters, kernel_size=3, strides=2, padding='same')(x)
    elif mode == 'max_pool':
        return MaxPooling2D()(x)
    elif mode == 'average_pool':
        return AveragePooling2D()(x)
    else:
        def resize_pic(x, h_f=0.5, w_f=0.5):
            K.resize_images(x, height_factor=h_f, width_factor=w_f, data_format='channels_last')
        return Lambda(lambda x: resize_pic(x))(x)


def subpixel(x, h_factor, w_factor):
    input_shape = tf.shape(x)
    batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3],
    output_channels = c // (h_factor * w_factor)
    new_x = K.reshape(x, shape=(batch_size, h, w, h_factor, w_factor, output_channels))
    new_x = K.permute_dimensions(new_x, (0, 1, 3, 2, 4, 5))
    output = K.reshape(new_x, shape=(batch_size, h*h_factor, w * w_factor, output_channels))
    return output


def residual_block(x, n_filters, kernel_size=3, strides=1, padding='same', activation='relu', sn=False):
    """
    修改说明：
    2019.10.13: 1.将return部分的Lambda层改为了Add层，原理一样
                2.添加了可选的SN层（SpectralNormalization）
    """
    if sn is True:
        residual1 = SpectralNormalization(Conv2D(filters=n_filters, 
                    kernel_size=kernel_size, strides=strides, padding=padding,
                    activation=activation))(x)
        residual2 = SpectralNormalization(Conv2D(filters=n_filters, 
                    kernel_size=kernel_size, strides=strides, padding=padding,
                    activation=activation))(residual1)
    else:
        residual1 = Conv2D(filters=n_filters, kernel_size=kernel_size,
                    strides=strides, padding=padding,
                    activation=activation)(x)
        residual1 = BatchNormalization()(residual1)
        residual2 = Conv2D(filters=n_filters, kernel_size=kernel_size,
                    strides=strides, padding=padding,
                    activation=activation)(residual1)
    return Add()([x, residual2])

def dense_block(x, n_filters=64):
    """
    这是一个5个 filters=64 的 Conv2D堆成的Dense Block，可根据需要修改
    NOTE 这个和Densenet的结构并不一致，实际上差别很大，请慎重使用
    """
    x1 = Conv2D(n_filters, kernel_size=3, strides=1, padding='same')(x)
    x1 = LeakyReLU(0.2)(x1)
    x1 = Concatenate()([x, x1])

    x2 = Conv2D(n_filters, kernel_size=3, strides=1, padding='same')(x1)
    x2 = LeakyReLU(0.2)(x2)
    x2 = Concatenate()([x, x1, x2])

    x3 = Conv2D(n_filters, kernel_size=3, strides=1, padding='same')(x2)
    x3 = LeakyReLU(0.2)(x3)
    x3 = Concatenate()([x, x1, x2, x3])

    x4 = Conv2D(n_filters, kernel_size=3, strides=1, padding='same')(x3)
    x4 = LeakyReLU(0.2)(x4)
    x4 = Concatenate()([x, x1, x2, x3, x4]) 

    x5 = Conv2D(n_filters, kernel_size=3, strides=1, padding='same')(x4)
    # x5 = Lambda(lambda x: x * 0.2)(x5)    # 超分中用到的
    x = Add()([x5, x])
    return x


def encode_block(x, n_filters, kernel_size=3, strides=1, padding='same'):
    conv = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
    bn = BatchNormalization()(conv)
    re = keras.layers.LeakyReLU(0.2)(bn)
    return re


def decode_block(x, n_filters, kernel_size=3, strides=1, padding='same'):
    x = Conv2D(filters=4 * n_filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               activation='relu')(x)
    out = Lambda(lambda z: tf.nn.depth_to_space(z, block_size=2))(x)
    return out


def d_block(input, filters, kernel_size, strides, padding):
    conv_d = Conv2D(filters=filters, kernel_size=kernel_size,
                  strides=strides, padding=padding)(input)
    norm_d = BatchNormalization()(conv_d)
    re_d = LeakyReLU(0.2)(norm_d)
    return re_d


def v_hourglass(a, p, n_filters, kernel_size, strides=1, padding='same'):
    x1 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=2*strides, padding=padding, activation='relu')(a)
    x2 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=2*strides, padding=padding, activation='relu')(x1)
    x3 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=2*strides, padding=padding, activation='relu')(x2)
    x4 = UpSampling2D(size=2)(x3)
    x4 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(x4)
    con1 = concatenate([x2, x4], axis=-1)
    x5 = UpSampling2D(size=2)(con1)
    x5 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(x5)
    con2 = concatenate([x1, x5], axis=-1)
    x6 = UpSampling2D(size=2)(con2)
    x6 = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu')(x6)
    return x6, None



