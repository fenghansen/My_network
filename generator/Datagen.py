from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

class Datagenerator(ImageDataGenerator):
    def __init__(self, *args, **kwargs):
        super(Datagenerator, self).__init__(self, *args, **kwargs)

    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, *args, **kwargs)
        while True:
            images = next(generator)
            images = images[0]
            for index, img in enumerate(images):
                images[index] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield np.array(images, dtype=np.uint8)



if __name__ == '__main__':
    dataPath = './data/imgs'
    dateGen = Datagenerator()
    dataIter = dateGen.flow_from_directory(dataPath)
    import cv2
    img = next(dataIter)
    cv2.imshow('window', img[0])
    cv2.waitKey(0)
