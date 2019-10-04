import keras
import keras.backend as K
from keras.layers import Conv2D, Lambda, UpSampling2D, concatenate, BatchNormalization
import tensorflow as tf


def downsampling(x, n_filters):
    return Conv2D(filters=n_filters, kernel_size=3, strides=2, padding='same')(x)


def subpixel(x, h_factor, w_factor):
    input_shape = tf.shape(x)
    batch_size, h, w, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3],
    output_channels = c // (h_factor * w_factor)
    new_x = K.reshape(x, shape=(batch_size, h, w, h_factor, w_factor, output_channels))
    new_x = K.permute_dimensions(new_x, (0, 1, 3, 2, 4, 5))
    output = K.reshape(new_x, shape=(batch_size, h*h_factor, w * w_factor, output_channels))
    return output


def residual_block(x, n_filters, kernel_size=3, strides=1, padding='same'):
    residual1 = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               activation='relu')(x)
    residual2 = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               activation='relu')(residual1)
    return Lambda(lambda z: z[0]+z[1])([x, residual2])


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
    re_d = keras.layers.LeakyReLU(0.2)(norm_d)
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



