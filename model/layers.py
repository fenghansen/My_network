import keras
import keras.backend as K
import tensorflow
from keras.layers import Conv2D, Activation


def downsampling(x, n_filters):
    return Conv2D(filters=n_filters, kernel_size=3, strides=2, padding='same')(x)

def subpixel(x, n_filters):
    pass

def residual_block(x, n_filters, kernel_size=3, strides = 1, padding = 'same'):
    residual = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               activation='relu')(x)
    residual = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               activation='relu')(residual)
    residual = Conv2D(filters=n_filters, kernel_size=kernel_size,
               strides=strides, padding=padding,
               activation='relu')(residual)
    return x + residual


def Encode_block(x, n_filters, kernel_size=3, strides = 1, padding = 'same'):
    x = residual_block(x, n_filters, kernel_size=kernel_size,
                       strides = strides, padding = padding)
    x = downsampling(x, n_filters)
    return x


def Decode_block(x, n_filters, kernel_size=3, strides = 1, padding = 'same'):
    x = residual_block(x, n_filters, kernel_size=kernel_size,
                       strides=strides, padding=padding)
    x = subpixel(x, n_filters)
    return x





