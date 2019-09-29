from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from model.layers import *
from model.loss import get_dloss, get_gloss



def build_g(config):
    # feature extraction
    y_hat = Input(shape=config['y_hat_shape'])
    x = Input(shape=config.x_shape)
    conv1 = encode_block(y_hat, n_filters=128, kernel_size=3, strides=1, padding='same')
    conv2 = encode_block(conv1, 128, 3, 1, padding='same')

    conv3 = encode_block(conv2, 128, 3, 1, padding='same')
    conv4 = encode_block(conv3, 128, 3, 1, padding='same')

    addition = build_f(config, x, y_hat)
    conv5 = keras.layers.concatenate([conv4, addition])
    up_sample = decode_block(conv5, None, 128, 3, 1, 'same')
    # decode
#    up_sample1 = Decode_block(conv8, conv7, n_filters=128, kernel_size=3, strides = 1, padding = 'same')

    up_sample1 = decode_block(conv3, up_sample, 128, 3, 1, 'same')

    up_sample2 = decode_block(up_sample1,conv2, 128, 3, 1, 'same')

    up_sample3 = decode_block(up_sample2, conv1, 128, 3, 1, 'same')

    up_sample4 = decode_block(up_sample3, input, 128, 3, 1, 'same')

    output = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(up_sample4)
    model = Model(inputs=input, outputs=[conv4, addition, output])
    return model


def build_stacked_hourglass(config):
    """
    build generator using stacked hourglass network
    :param config: network configuration
    :return:  generative model
    """
    img_height = config['img_height']
    img_width = config['img_width']
    channels = config['img_channels']
    p_height = config['p_height']
    p_width = config['p_width']
    p_channels = config['p_channels']

    input_img = Input(shape=(img_height, img_width, channels))
    input_pose = Input(shape=(p_height, p_width, p_channels))

    a1, p1 = v_hourglass(input_img, input_pose, 64, 3)
    a2, p2 = v_hourglass(a1, p1, 64, 3)
    a3, p3 = v_hourglass(a2, p2, 64, 3)

    output = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(a3)
    model = Model(inputs=[input_img, input_pose], outputs=[output])
    return model


def build_d(config):
    """
    discriminator of gan
    :param config: store the network configuration
    :return: discriminator model
    """
    img_height = config['img_height']
    img_width = config['img_width']
    channels = config['img_channels']

    input_img = Input(shape=(img_height, img_width, channels))

    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(input_img)
    x = residual_block(x, n_filters=64, kernel_size=3, strides=1, padding='same')
    x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = residual_block(x, n_filters=64, kernel_size=3, strides=1, padding='same')
    output = Conv2D(filters=3, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    score = Lambda(lambda z: K.mean(K.tanh(z)))(output)
    model = Model(inputs=input_img, outputs=score)
    return model


def build_f(config, x, y_hat):
    input = keras.layers.concatenate([x, y_hat])
    conv1 = encode_block(input, n_filters=128, kernel_size=3, strides=1, padding='same')
    conv2 = encode_block(conv1, 128, 3, 1, padding='same')

    conv3 = encode_block(conv2, 128, 3, 1, padding='same')
    conv4 = encode_block(conv3, 128, 3, 1, padding='same')
    return conv4


def build_gan(g_net, d_net, config):
    """
    define the loss function of discriminator and
    generator and build training model
    :return: discriminator training model d_net and generator trainging model g_net
    """
    img_height = config['img_height']
    img_width = config['img_width']
    channels = config['img_channels']
    p_height = config['p_height']
    p_width = config['p_width']
    p_channels = config['p_channels']

    x_in = Input(shape=(img_height, img_width, channels))
    y_hat = Input(shape=(p_height, p_width, p_channels))
    gt = Input(shape=(img_height, img_width, channels))

    g_net.trainable = False

    x_fake = g_net([x_in, y_hat])
    x_real = gt

    real_score = d_net(x_real)
    fake_score = d_net(x_fake)

    d_train_model = Model(inputs=[x_in, y_hat, gt], outputs=real_score)

    # eps = np.random.rand()
    # x_inter = x_real * eps + (1.-eps) * x_fake
    # grad = K.gradients(tf.reduce_mean(d_net(x_inter)), [x_inter])[0]
    # grad_norm = K.sqrt(tf.reduce_mean(grad**2))
    # gradient_penalty = 10 * tf.reduce_mean(K.relu(1. - grad_norm))
    gradient_penalty = 0

    d_loss = get_dloss(real_score, fake_score)
    d_train_model.add_loss(K.mean(d_loss))
    d_train_model.compile(optimizer=Adam(lr=2e-5))

    g_net.trainable = True
    d_net.trainable = False

    g_train_model = Model(inputs=[x_in, y_hat, gt], outputs=fake_score)
    g_loss = get_gloss(fake_score, x_fake, x_real)
    g_train_model.add_loss(K.mean(g_loss))
    g_train_model.compile(optimizer=Adam(lr=2e-5))
    return g_train_model, d_train_model
