from model import *
from keras.layers import Input
import yaml
from generator import Dategen

if __name__ == '__main':

    config = yaml.load('../config/config.yaml')


    G_net = build_G()
    D_net = build_D()
    x_input = Input(shape=(config.input_shape))
    x_real = Input(shape=(config.input_shape))

    for epoch in config.Epochs:
        for step in config.Steps:

