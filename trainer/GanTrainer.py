from generator import Dategen

class Gan_Trainer(object):
    def __init__(self, G_net, D_net, directory = None, preprocess = True,
                 re_train = True, load_dir = None, save_dir = None):
        self.g_model = G_net
        self.d_model = D_net
        self.train_dir = directory
        self.preprocess = preprocess
        self.re_train = re_train
        self.load_dir = load_dir
        self.save_dir = save_dir

    def train(self):
        # train using image_generator
        if self.preprocess == True:
            train_datagen = Dategen()
            train_generator = train_datagen.flow_from_directory()

