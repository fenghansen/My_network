from keras.preprocessing.image import ImageDataGenerator

class Datagenerator(ImageDataGenerator):
    def __init__(self):
        super(Datagenerator, self).__init__()

    def flow_from_directory(self,**kwargs):
        generator = self.flow_from_directory(**kwargs)
        while 1:
            image = next(generator)
            yield image



