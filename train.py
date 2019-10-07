from model.network import build_g, build_d, build_gan, build_res
import yaml
from trainer.GanTrainer import Trainer
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

if __name__ == '__main__':
    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    g_net = build_res(config)
    d_net = build_d(config)

    load_weights = True
    if load_weights:
        g_net.load_weights(config['load_dir'] + 'g_model.h5')
        d_net.load_weights(config['load_dir'] + 'd_model.h5')

    g_train_model, d_train_model = build_gan(g_net, d_net, config)

    g_net.summary()
    d_net.summary()
    g_train_model.summary()
    d_train_model.summary()

    trainer = Trainer(config, g_net, d_net, g_train_model, d_train_model, train_on_gan=True)
    if not os.path.exists('/media/newbot/software/Pose-Transfer-master/fashion_data'):
        raise Exception
    trainer.train('/media/newbot/software/Pose-Transfer-master/fashion_data/train/',
                   '/media/newbot/software/Pose-Transfer-master/fashion_data/trainK/',
                  '/media/newbot/software/Pose-Transfer-master/fashion_data/fasion-resize-pairs-train.csv')

