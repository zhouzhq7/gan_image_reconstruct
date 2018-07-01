import argparse
import time
import tensorlayer as tl
from god_config import config
from utils import *
from model import VGG19, generator, discriminator
import numpy as np
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--image_raw_data_dir', type=str, default=config.raw_image_dir,
                    help='directory contains all raw images with size (224, 224, 3)')

parser.add_argument('--mode', type=str, default='train',
                    help='train or evaluate')
"optimizer"
batch_size = config.batch_size
lr_init = config.lr_init
beta1 = config.beta1

"initialize g"
n_epoch_init = config.n_epoch_init

"adversarial learning (GAN)"
n_epoch = config.n_epoch
lr_decay = config.lr_decay
decay_every = config.decay_every

"tfrecord data file"
filename = config.data_tfrecord_dir

def train():
    test_images = get_test_images()

    # <editor-fold desc="build tensorflow graph">
    # Build model

    t_image = tf.placeholder(dtype='float32', shape=[None, 224, 224, 3],
                             name='t_image_input_to_vgg19')

    net_vgg, inputs_g = VGG19((t_image/255.0), reuse=False)

    net_g, _ = generator(inputs_g, is_train=True, reuse=False)

    net_d, logits_real = discriminator((t_image/127.5)-1.0, is_train=True, reuse=False)

    _, logits_fake = discriminator(net_g.outputs, is_train=True, reuse=True)

    # ??? can i define test network as following?
    net_g_test, _ = generator(inputs_g, is_train=False, reuse=True)
    ## define loss
    "generator"
    d_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real,
                                                      labels=tf.ones_like(logits_real), name='d_loss1')

    d_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
                                                      labels=tf.zeros_like(logits_fake), name='d_loss2')

    d_loss = d_loss1 + d_loss2

    "discriminator"

    g_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake,
                                                      labels=tf.ones_like(logits_fake), name='g_loss1')

    mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(net_g.outputs, (t_image/127.5)-1.0))

    g_loss = g_loss1 + mse_loss

    g_vars = tl.layers.get_variables_with_name(name='generator', train_only=True, printable=True)
    d_vars = tl.layers.get_variables_with_name(name='discriminator', train_only=True, printable=True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    "pretrain generator"
    g_optim_init = tf.train.AdamOptimizer(learning_rate=lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)

    "gan"
    g_optim = tf.train.AdamOptimizer(learning_rate=lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    d_optim = tf.train.AdamOptimizer(learning_rate=lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    # </editor-fold>


    # <editor-fold desc="make directories ">
    save_ginit_dir = "./samples/{}_ginit".format(tl.global_flag['mode'])
    save_gan_dir = "./samples/{}_gan".format(tl.global_flag['mode'])
    checkpoints_dir = "./checkpoints"
    pre_trained_model_dir = "./models"

    mkdir_if_not_exists(save_gan_dir)
    mkdir_if_not_exists(save_ginit_dir)
    mkdir_if_not_exists(checkpoints_dir)
    mkdir_if_not_exists(pre_trained_model_dir)
    # </editor-fold>

    # <editor-fold desc="restore model">
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    tl.layers.initialize_global_variables(sess)

    if tl.files.load_and_assign_npz(sess=sess,
                                    name=checkpoints_dir+"/g_{}.npz".format(tl.global_flag['mode']),
                                    network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess,
                                     name=checkpoints_dir + "/g_{}_init.npz".format(tl.global_flag['mode']),
                                     network=net_g)

    tl.files.load_and_assign_npz(sess=sess,
                                 name=checkpoints_dir+"d_{}.npz".format(tl.global_flag['mode']),
                                 network=net_d)
    # </editor-fold>

    # <editor-fold desc="load vgg 19 model">
    vgg19_model_path = pre_trained_model_dir+'/vgg19.npy'

    if not os.path.isfile(vgg19_model_path):
        raise Exception("Cannot find vgg19 model.")

    npz = np.load(vgg19_model_path, encoding='latin1').item()

    params = []

    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # </editor-fold>

    sess.run(tf.assign(lr_v, lr_init))

    print ("Initializing G with learning rate {}".format(lr_init))

    "get iterator to get image batch from pre-stored tfrecord file"
    img_batch_g_init = inputs(filename, batch_size, n_epoch_init,
                       shuffle_size=10000, is_augment=True)

    num_of_data = 45611
    num_of_iter_one_epoch = num_of_data // batch_size
    try:
        epoch_time = time.time()
        total_mse_loss, n_iter = 0.0 , 0

        while True:
            if (n_iter + 1) % num_of_iter_one_epoch == 0:
                log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
                epoch_time = time.time()
                print (log)
            if ((n_iter+1) % (10*num_of_iter_one_epoch) == 0):
                tl.files.save_npz(net_g.all_params,
                                  name=checkpoints_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
            step_time = time.time()
            imgs = sess.run(img_batch_g_init)
            err, _ = sess.run([mse_loss, g_optim_init], feed_dict={t_image:imgs.astype(np.float32)})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
                (n_iter + 1) // num_of_iter_one_epoch, n_epoch_init, n_iter, time.time() - step_time, err))
            total_mse_loss += err
            n_iter += 1

    except tf.errors.OutOfRangeError:
        tl.files.save_npz(net_g.all_params,
                          name=checkpoints_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)
        print ("Done initializing G.")


    #################################train gan##################################
    img_batch_gan = inputs(filename, batch_size=batch_size, num_epochs=n_epoch,
                           shuffle_size=10000, is_augment=True)

    try:
        epoch_time = time.time()
        total_g_loss, total_d_loss, n_iter = 0, 0, 0

        while True:
            if (n_iter + 1) == 1:
                sess.run(tf.assign(lr_v, lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
                print(log)

            elif (n_iter+1) % (decay_every * num_of_iter_one_epoch) == 0:
                new_lr_decay = lr_decay ** ((n_iter+1) // (decay_every * num_of_iter_one_epoch))
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)

            if ((n_iter + 1) % num_of_iter_one_epoch == 0):
                log = "[*] Epoch [%4d/%4d] time: %4.4fs, d_loss: %8f, g_loss: %8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch, time.time()-epoch_time, total_d_loss/num_of_iter_one_epoch,
                    total_g_loss/num_of_iter_one_epoch
                )
                print (log)
                total_g_loss, total_d_loss = 0, 0
                epoch_time = time.time()

            step_time = time.time()
            imgs = sess.run(img_batch_gan)
            "update discriminator"
            err_d, _ = sess.run([d_loss, d_optim], feed_dict={t_image: imgs})
            "update generator"
            err_g, err_mse, err_gan_loss, _ = sess.run([g_loss, mse_loss, g_loss1, g_optim],
                                                       feed_dict={t_image: imgs})

            log = "Epoch [%4d/%4d] %6d time: %4.4fs, d_loss: %8f, g_loss: %8f, (mse: %8f, gan_loss: %8f)" % (
                (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_d, err_g,
                err_mse, err_gan_loss
            )

            print (log)


            total_d_loss += err_d
            total_g_loss += err_g
            ## quick evaluation on train set
            if ( (n_iter + 1) % (num_of_iter_one_epoch * 10) == 0):
                out = sess.run(net_g_test.outputs,
                               {t_image: test_images})  # ; print('gen sub-image:', out.shape, out.min(), out.max())
                out = (out+1)*127.5
                print ("gen sub image:", out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out.astype(np.uint8), [8, 8], save_gan_dir + '/train_%d.png' % ((n_iter + 1) // num_of_iter_one_epoch))

            ## save model
            if ( (n_iter + 1) % (num_of_iter_one_epoch * 10) == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoints_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                                  sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoints_dir + '/d_{}.npz'.format(tl.global_flag['mode']),
                                  sess=sess)
            n_iter += 1




    except tf.errors.OutOfRangeError:
        tl.files.save_npz(net_g.all_params,
                          name=checkpoints_dir+'/g_{}.npz'.format(tl.global_flag['mode'], sess=sess))
        tl.files.save_npz(net_d.all_params,
                          name=checkpoints_dir+'/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)


def evaluate():
    pass

if __name__ == '__main__':
    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    else:
        raise Exception('Unknow mode {}'.format(args.mode))


