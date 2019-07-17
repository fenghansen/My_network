import keras.backend as K
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.models import Model

def Vgg_loss(x):
    base_model = VGG19(weights='imagenet', include_top=False)
    feature_layers = [
        "input_1",
        "block1_conv2", "block2_conv2",
        "block3_conv2", "block4_conv2",
        "block5_conv2"]
    feature = [base_model.get_layer(k).output for k in feature_layers]
    model = Model(inputs=base_model.input, outputs=feature)
    return model.predict(x)

def g_loss(y_pred):
    p_zy = y_pred[0]
    q_xy = y_pred[1]
    p_yz = y_pred[2]
    kl_loss = tf.reduce_sum(q_xy * tf.log(q_xy) - p_zy * tf.log(p_zy))
    perceptual_loss = Vgg_loss(p_yz)
    return kl_loss + perceptual_loss

def d_loss():
    pass