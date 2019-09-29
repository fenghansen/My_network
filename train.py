from model.network import build_g, build_d, build_gan, build_stacked_hourglass
import yaml
from trainer.GanTrainer import Trainer
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

if __name__ == '__main__':
    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    g_net = build_stacked_hourglass(config)
    d_net = build_d(config)
    g_model, d_model = build_gan(g_net, d_net, config)
    trainer = Trainer(config, g_model, d_model, train_on_gan=True)
    if not os.path.exists('/media/newbot/software/Pose-Transfer-master/fashion_data'):
        raise Exception
    trainer.train('/media/newbot/software/Pose-Transfer-master/fashion_data/train/',
                   '/media/newbot/software/Pose-Transfer-master/fashion_data/trainK/')
