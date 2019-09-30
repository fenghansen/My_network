from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from model.layers import *
from model.loss import get_dloss, get_gloss



def build_g(config):
    # feature extraction
    img_height = config['img_height']
    img_width = config['img_width']
    channels = config['img_channels']
    p_height = config['p_height']
    p_width = config['p_width']
    p_channels = config['p_channels']

    input_img = Input(shape=(img_height, img_width, channels))
    input_pose = Input(shape=(p_height, p_width, p_channels))

    network_input = keras.layers.concatenate([input_img, input_pose])

    conv1 = encode_block(network_input, n_filters=64, kernel_size=3, strides=1, padding='same')
    conv2 = encode_block(conv1, 128, 3, 1, padding='same')

    conv3 = encode_block(conv2, 256, 3, 1, padding='same')
    conv4 = encode_block(conv3, 512, 3, 1, padding='same')

    conv5 = encode_block(conv4, 512, 3, 1, padding='same')

    conv6 = encode_block(conv5, 512, 3, 1, padding='same')
    up = UpSampling2D(size=2)(conv6)

    con1 = concatenate([up, conv5], axis=-1)
    up2 = UpSampling2D(size=2)(con1)
    up2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(up2)

    con2 = concatenate([up2, conv4], axis=-1)
    up3 = UpSampling2D(size=2)(con2)
    up3 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(up3)

    con3 = concatenate([up3, conv3], axis=-1)
    up4 = UpSampling2D(size=2)(con3)
    up4 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')(up4)

    con4 = concatenate([up4, conv2], axis=-1)
    up5 = UpSampling2D(size=2)(con4)
    up5 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')(up5)

    con5 = concatenate([up5, conv1], axis=-1)
    up6 = UpSampling2D(size=2)(con5)
    up6 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(up6)

    output = Conv2D(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(up6)
    model = Model(inputs=[input_img, input_pose], outputs=[output])
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
    x = residual_block(x, n_filters=128, kernel_size=3, strides=1, padding='same')
    x = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = residual_block(x, n_filters=512, kernel_size=3, strides=1, padding='same')
    output = Conv2D(filters=3, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    score = Lambda(lambda z: K.mean(K.sigmoid(z)))(output)
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
    d_train_model.compile(optimizer=Adam(lr=2e-4))

    g_net.trainable = True
    d_net.trainable = False

    g_train_model = Model(inputs=[x_in, y_hat, gt], outputs=fake_score)
    g_loss = get_gloss(fake_score, x_fake, x_real)
    g_train_model.add_loss(K.mean(g_loss))
    g_train_model.compile(optimizer=Adam(lr=2e-4))
    return g_train_model, d_train_model
