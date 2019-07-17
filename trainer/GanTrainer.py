from generator import Dategen

class Gan_Trainer():
    def __init__(self, config, g_model, d_model, preprocess = True, re_train = True, train_on_gan = False):
        self.g_model = g_model
        self.d_model = d_model
        self.preprocess = preprocess
        self.re_train = re_train
        self.config = config
        self.train_on_gan = train_on_gan

    def train(self):
        # train using image_generator
        if self.preprocess == True:
            train_datagen = Dategen()
            train_generator = train_datagen.flow_from_directory()
            if self.train_on_gan:
                # load pre_retained model
                if not self.re_train:
                    self.g_model.load_weights(self.config.load_dir + 'g_model.h5')
                    self.d_model.load_weights(self.config.load_dir + 'd_model.h5')
                for epoch in self.config.Epochs:
                    for step in self.config.Steps:
                        img, pose_img = next(train_generator)
                        x_real = img
                        x_fake = self.g_model.predict(x_real)
                        g_loss = self.g_model.train_on_batch([pose_img, x_fake, x_real])
                        d_loss = self.d_model.train_on_batch([x_real, x_fake])
                        print("Epoch: {}, steps: {}, g_loss: {}, d_loss: {}".format(epoch, step, g_loss, d_loss))
            else:
                # load pre_retained model
                if not self.re_train:
                    self.g_model.load_weights(self.config.load_dir + 'g_model.h5')
                for epoch in self.config.Epochs:
                    for step in self.config.Steps:
                        img, pose_img = next(train_generator)
                        x_real = img
                        x_fake = self.g_model.predict(x_real)
                        g_loss = self.g_model.train_on_batch([pose_img, x_fake, x_real])
                        print("Epoch: {}, steps: {}, g_loss: {}".format(epoch, step, g_loss))
        else:
            pass

    def save(self):
        self.g_model.save(self.config.save_dir + 'g_model.h5')
        self.d_model.save(self.config.save_dir + 'd_model.h5')
        print("Model saved Successfully!")

