from model.network import build_g
from generator.dataIter import DataIter
import cv2
import yaml
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '0'

if __name__ == '__main__':
    with open('config/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    g_net = build_g(config)
    g_net.summary()
    g_net.load_weights(config['load_dir'] + 'g_model.h5')
    img_dir = "/media/newbot/software/Pose-Transfer-master/fashion_data/test/"
    keypoint_dir = "/media/newbot/software/Pose-Transfer-master/fashion_data/testK/"
    dataIter = DataIter(img_dir, keypoint_dir, 1, True)
    test_gen = dataIter.gen_next()
    img, pose_img, target_img, target_pose = next(test_gen)
    predicted = g_net.predict([img, target_pose])
    cv2.imshow('pred', predicted[0])
    cv2.imshow('img', img[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()