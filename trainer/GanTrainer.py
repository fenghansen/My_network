from generator.dataIter import DataIter


class Trainer:
    def __init__(self, config, g_model, d_model, preprocess=True, re_train=True, train_on_gan=False,
                 batch_size=6):
        self.g_model = g_model
        self.d_model = d_model
        self.preprocess = preprocess
        self.re_train = re_train
        self.config = config
        self.train_on_gan = train_on_gan
        self.batch_size = batch_size

    def train(self, img_dir, keypoint_dir):
        # train using image_generator
        if self.preprocess == True:
            dataIter = DataIter(img_dir, keypoint_dir, self.batch_size, True)
        else:
            dataIter = DataIter(img_dir, keypoint_dir, self.batch_size, False)
        train_generator = dataIter.gen_next()
        if self.train_on_gan:
            # load pre_retained model
            if not self.re_train:
                self.g_model.load_weights(self.config['load_dir'] + 'g_model.h5')
                self.d_model.load_weights(self.config['load_dir'] + 'd_model.h5')
            for epoch in range(self.config['epochs']):
                for step in range(self.config['steps']):
                    img, pose_img, target_img, target_pose = next(train_generator)
                    x_real = target_img
                    # x_fake = self.g_model.predict(img, target_pose)
                    g_loss = self.g_model.train_on_batch([img, target_pose, x_real], None)
                    d_loss = self.d_model.train_on_batch([img, target_pose, x_real], None)
                    print("Epoch: {}, steps: {}, g_loss: {}, d_loss: {}".format(epoch, step, g_loss, d_loss))
        else:
            # load pre_retained model
            if not self.re_train:
                self.g_model.load_weights(self.config['load_dir'] + 'g_model.h5')
            for epoch in self.config['epochs']:
                for step in self.config['steps']:
                    img, pose_img, target_img, target_pose = next(train_generator)
                    x_real = img
                    # x_fake = self.g_model.predict(x_real)
                    g_loss = self.g_model.train_on_batch([img, target_pose, x_real], x_real)
                    print("Epoch: {}, steps: {}, g_loss: {}".format(epoch, step, g_loss))

    def save(self):
        self.g_model.save(self.config.save_dir + 'g_model.h5')
        self.d_model.save(self.config.save_dir + 'd_model.h5')
        print("Model saved Successfully!")

