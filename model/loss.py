from keras.applications.vgg19 import VGG19
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf


def l1_norm(m1, m2):
    if K.ndim(m1) == 4:
        return K.sum(K.abs(m1 - m2), axis=[1, 2, 3])
    elif K.ndim(m1) == 3:
        return K.sum(K.abs(m1 - m2), axis=[1, 2])


def Vgg_loss(x_fake, x_real):
    base_model = VGG19(weights='imagenet', include_top=False)
    feature_layers = [
        "block1_conv2", "block2_conv2",
        "block3_conv2", "block4_conv2",
        "block5_conv2"]
    feature = [base_model.get_layer(k).output for k in feature_layers]
    vgg = Model(inputs=base_model.input, outputs=feature)
    vgg.trainable = False
    vgg.compile(loss='mse', optimizer='adam')
    pred = np.array(vgg(x_fake))
    gt = np.array(vgg(x_real))
    loss = 0
    N = K.cast(K.shape(x_fake)[0], dtype=tf.float32)
    for p, g in zip(pred, gt):
        loss += l1_norm(p, g) / N
    return loss


def get_gloss(fake_score, x_fake, x_real):
    # p_zy = y_pred[0]
    # q_xy = y_pred[1]
    # p_yz = y_pred[2]
    # kl_loss = tf.reduce_sum(q_xy * tf.log(q_xy) - p_zy * tf.log(p_zy))
    adverarial_loss = -K.log(1 - fake_score + 1e-9)
    perceptual_loss = K.mean(Vgg_loss(x_fake, x_real))
    l1 = K.mean(l1_norm(x_fake, x_real))
    return K.mean(adverarial_loss) + 0.05 * perceptual_loss +  l1


def get_dloss(real_score, fake_score):
    return K.mean(-K.log(real_score + 1e-9) - K.log(1 - fake_score + 1e-9))
