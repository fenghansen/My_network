from generator.dataIter import DataIter
import cv2

class Trainer:
    def __init__(self, config, g_net, d_net, g_train_model, d_train_model, preprocess=True, train_on_gan=False,
                 batch_size=6):
        self.g_net = g_net
        self.d_net = d_net
        self.g_model = g_train_model
        self.d_model = d_train_model
        self.preprocess = preprocess
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
            for epoch in range(self.config['epochs']):
                for step in range(self.config['steps']):
                    g_loss = 0
                    img, pose_img, target_img, target_pose = next(train_generator)
                    x_real = target_img
                    d_loss = self.d_model.train_on_batch([img, target_pose, x_real], None)
                    for i in range(2):
                        g_loss = self.g_model.train_on_batch([img, target_pose, x_real], None)
                        img, pose_img, target_img, target_pose = next(train_generator)
                        x_real = target_img
                    print("Epoch: {}, steps: {}, g_loss: {}, d_loss: {}".format(epoch, step, g_loss, d_loss))
                print("saving model......")
                self.save()
                test_img, test_pose_img, test_target_img, test_target_pose = next(train_generator)
                predicted = self.g_net.predict([test_img, test_target_pose])
                cv2.imshow('window', predicted[0])
                cv2.waitKey(800)
                cv2.imshow('window2', test_target_img[0])
                cv2.waitKey(800)
                cv2.imshow('window3', test_img[0])
                cv2.waitKey(800)
                cv2.destroyAllWindows()
        else:
            for epoch in self.config['epochs']:
                for step in self.config['steps']:
                    img, pose_img, target_img, target_pose = next(train_generator)
                    x_real = img
                    # x_fake = self.g_model.predict(x_real)
                    g_loss = self.g_model.train_on_batch([img, target_pose, x_real], None)
                    print("Epoch: {}, steps: {}, g_loss: {}".format(epoch, step, g_loss))

    def save(self):
        self.g_net.save(self.config['save_dir'] + 'g_model.h5')
        self.d_net.save(self.config['save_dir'] + 'd_model.h5')
        print("Model saved Successfully!")

