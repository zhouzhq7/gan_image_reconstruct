from easydict import EasyDict as edict

config = edict()


"data and check point directories"
config.raw_image_dir = ''
config.data_tfrecord_dir = './train/images.tf'


"optimization"
config.batch_size = 64
config.lr_init = 1e-5
config.beta1 = 0.9

"g initialization"
config.n_epoch_init = 0

config.n_epoch = 2000
config.lr_decay = 0.1
config.decay_every = int(config.n_epoch/2)
