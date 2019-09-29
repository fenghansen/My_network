import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.models import Model
import keras.backend as K
import numpy as np

def Vgg_loss(x_fake, x_real):
    base_model = VGG19(weights='imagenet', include_top=False)
    base_model.summary()
    feature_layers = [
        "input_7",
        "block1_conv2", "block2_conv2",
        "block3_conv2", "block4_conv2",
        "block5_conv2"]
    feature = [base_model.get_layer(k).output for k in feature_layers]
    model = Model(inputs=base_model.input, outputs=feature)
    pred = np.array(model(x_fake))
    gt = np.array(model(x_real))
    loss = 0
    for p, g in zip(pred, gt):
        loss += K.mean(p - g)
    return loss


def get_gloss(fake_score, x_fake, x_real):
    # p_zy = y_pred[0]
    # q_xy = y_pred[1]
    # p_yz = y_pred[2]
    # kl_loss = tf.reduce_sum(q_xy * tf.log(q_xy) - p_zy * tf.log(p_zy))
    adverarial_loss = tf.log(1 - fake_score)
    perceptual_loss = Vgg_loss(x_fake, x_real)
    return adverarial_loss + perceptual_loss


def get_dloss(real, fake):
    return tf.log(real) + tf.log(1 - fake)
