import pandas as pd
import random
import cv2
import numpy as np


class DataIter:
    def __init__(self, img_dir, keypoint_dir, csv_dir, batch_size=32):
        self.img_dir = img_dir
        self.keypoint_dir = keypoint_dir
        self.file_set = pd.read_csv(csv_dir)
        self.batch_size = batch_size

    def gen_next(self):
        froms = self.file_set['from'].values
        tos = self.file_set['to'].values
        length = len(froms)
        while True:
            file_index = random.sample(range(length), self.batch_size)
            imgs = []
            keypoints = []
            target_imgs = []
            target_kpnts = []
            for index in file_index:
                img = np.array(cv2.imread(self.img_dir + froms[index]))
                keypoint = np.load(self.keypoint_dir + froms[index] + '.npy')
                imgs.append(img)
                keypoints.append(keypoint)
                target_img = np.array(cv2.imread(self.img_dir + tos[index]))
                target_k = np.load(self.keypoint_dir + tos[index] + '.npy')
                target_imgs.append(target_img)
                target_kpnts.append(target_k)
            input_img = np.array(imgs, dtype=np.float32) * 1./255
            input_k = np.array(keypoints, dtype=np.float32)
            target_imgs = np.array(target_imgs, dtype=np.float32) * 1./255
            target_kpnts = np.array(target_kpnts, dtype=np.float32)
            yield input_img, input_k, target_imgs, target_kpnts


if __name__ == '__main__':
    dataIter = DataIter('./data/imgs/test/', './data/keypoints/test/', './data/fasion-resize-pairs-test.csv')
    generator = dataIter.gen_next()
    input_img, input_k, target_img, target_k = next(generator)
    print(input_img.shape, input_k.shape)
    print(target_img.shape, target_k.shape)
    cv2.imshow('input', input_img[0])
    cv2.imshow('target', target_img[0])
    cv2.waitKey(0)