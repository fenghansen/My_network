import os
import random
import cv2
import numpy as np


class DataIter:
    def __init__(self, img_dir, keypoint_dir, batch_size=32, preprocess=True):
        self.img_dir = img_dir
        self.keypoint_dir = keypoint_dir
        self.file_set = os.listdir(self.img_dir)
        self.batch_size = batch_size

    def gen_next(self):
        while True:
            file_names = random.sample(self.file_set, k=self.batch_size)
            imgs = []
            keypoints = []
            for file in file_names:
                img = np.array(cv2.imread(self.img_dir + file))
                keypoint = np.load(self.keypoint_dir + file + '.npy')
                imgs.append(img)
                keypoints.append(keypoint)
            for file in file_names:
                target_set = [name for name in self.file_set if name.startswith(file[:file.rfind('_')])]
                new_name = random.choice(target_set)
                img = np.array(cv2.imread(self.img_dir + new_name))
                keypoint = np.load(self.keypoint_dir + new_name + '.npy')
                imgs.append(img)
                keypoints.append(keypoint)
            normalized = np.array(imgs, dtype=np.float32)
            normalized = normalized * 1./255
            input_img = normalized[:self.batch_size, :, :, :]
            input_k = np.array(keypoints)[:self.batch_size, :, :, :]
            target_img = normalized[self.batch_size:, :, :, :]
            target_k = np.array(keypoints)[self.batch_size:, :, :, :]
            yield input_img, input_k, target_img, target_k


if __name__ == '__main__':
    dataIter = DataIter('./data/imgs/test/', './data/keypoints/test/')
    generator = dataIter.gen_next()
    input_img, input_k, target_img, target_k = next(generator)
    print(input_img.shape, input_k.shape)
    print(target_img.shape, target_k.shape)
    cv2.imshow('input', input_img[0])
    cv2.imshow('target', target_img[0])
    cv2.waitKey(0)