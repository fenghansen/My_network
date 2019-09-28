from model.network import build_G, build_D, build_gan
import yaml
from trainer.GanTrainer import Gan_Trainer
import keras


if __name__ == '__main':
    config = yaml.load('../config/config.yaml')
    G_net = build_G(config)
    G_net.compile(optimizer=keras.optimizers.Adam(lr=2e-4))
    # D_net = build_D()
    # g_model, d_model = build_gan(G_net, D_net)
    trainer = Gan_Trainer(config, G_net, None)
    trainer.train()


