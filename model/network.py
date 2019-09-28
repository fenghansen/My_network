from layers import *
from keras.layers import Input
import tensorflow as tf
from keras.models import Model
from keras.optimizers import Adam

def build_G(config):
    # feature extraction
    y_hat = Input(shape=config.y_hat_shape)
    x = Input(shape=config.x_shape)
    conv1 = Encode_block(y_hat, n_filters=128, kernel_size=3, strides=1, padding='same')
    conv2 = Encode_block(conv1, 128, 3, 1, padding='same')

    conv3 = Encode_block(conv2, 128, 3, 1, padding='same')
    conv4 = Encode_block(conv3, 128, 3, 1, padding='same')

    addition = build_F(config, x, y_hat)
    conv5 = keras.layers.concatenate([conv4, addition])
    up_sample = Decode_block(conv5, None, 128, 3, 1, 'same')
    # decode
#    up_sample1 = Decode_block(conv8, conv7, n_filters=128, kernel_size=3, strides = 1, padding = 'same')

    up_sample1 = Decode_block(conv3, up_sample, 128, 3, 1, 'same')

    up_sample2 = Decode_block(up_sample1,conv2, 128, 3, 1, 'same')

    up_sample3 = Decode_block(up_sample2, conv1, 128, 3, 1, 'same')

    up_sample4 = Decode_block(up_sample3, input, 128, 3, 1, 'same')

    output = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(up_sample4)
    model = Model(inputs=input, outputs=[conv4, addition, output])
    return model

def build_stacked_hourglass(config_hg):
    img_height = config_hg['height']
    img_width = config_hg['width']
    channels = config_hg['channels']
    p_height = config_hg['p_height']
    p_width = config_hg['p_width']
    p_channels = config_hg['p_channels']

    input_img = Input(shape=(img_height, img_width, channels))
    input_pose = Input(shape=(p_height, p_width, p_channels))

    a1, p1 = v_hourglass(input_img, input_pose, 64, 3)
    a2, p2 = v_hourglass(a1, p1, 64, 3)
    a3, p3 = v_hourglass(a2, p2, 64, 3)
    a4, p4 = v_hourglass(a3, p3, 64, 3)

    output = Conv2D(3, 3, 1, 'same')(a4)
    model = Model(inputs=[input_img, input_pose], outputs=[output])
    return model


def build_D(config):
    pass


def build_F(config, x, y_hat):
    input = keras.layers.concatenate([x, y_hat])
    conv1 = Encode_block(input, n_filters=128, kernel_size=3, strides=1, padding='same')
    conv2 = Encode_block(conv1, 128, 3, 1, padding='same')

    conv3 = Encode_block(conv2, 128, 3, 1, padding='same')
    conv4 = Encode_block(conv3, 128, 3, 1, padding='same')
    return conv4


def build_gan(g_net, d_net):
    """
    define the loss function of discriminator and
    generator and build training model
    :return: discriminator training model d_net and generator trainging model g_net
    """
    x_in = Input(shape=(512, 512, 3))
    y_hat = Input(shape=(512, 512, 3))

    g_net.trainable = False

    x_fake = g_net([x_in, y_hat])
    x_real = x_in

    real_score = d_net(x_real)

    d_train_model = Model(inputs=[x_in, y_hat], outputs=real_score)

    # eps = np.random.rand()
    # x_inter = x_real * eps + (1.-eps) * x_fake
    # grad = K.gradients(tf.reduce_mean(d_net(x_inter)), [x_inter])[0]
    # grad_norm = K.sqrt(tf.reduce_mean(grad**2))
    # gradient_penalty = 10 * tf.reduce_mean(K.relu(1. - grad_norm))
    gradient_penalty = 0
    d_loss = 0
    d_train_model.add_loss(K.mean(d_loss))
    d_train_model.compile(optimizer=Adam(2e-3, 0.5))

    g_net.trainable = True
    d_net.trainable = False
    fake_score = d_net(x_fake)

    g_train_model = Model(inputs=[x_in, y_hat], outputs=fake_score)
    g_loss = 0
    g_train_model.add_loss(K.mean(g_loss))
    g_train_model.compile(optimizer=Adam(2e-4, 0.5))
    return d_train_model, g_train_model