from tensorpack import *
from tensorpack.dataflow.imgaug.base import ImageAugmentor
import tensorflow as tf
import argparse
import numpy as np
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
import cv2

from deepsense import neptune
context = neptune.Context()
context.integrate_with_tensorflow()

DEBUG = True

BATCH_SIZE = 32

class Scale(ImageAugmentor):
    """     Scale pixelvalues     """
    def __init__(self, input_domain=[0, 255], output_domain=[-1, 1]):
        self.input_domain = input_domain
        self.output_domain = output_domain

    @staticmethod
    def do(img, domain, range):
        # project into [0, 1]
        img = (img - domain[0]) / float(domain[1] - domain[0])
        # project into output_range
        img = img * (range[1] - range[0]) + float(range[0])
        return img

    def _augment(self, img, _):
        img = img.astype('float32')
        return Scale.do(img, self.input_domain, self.output_domain)

class CenterCropMax(ImageAugmentor):
    """ Crop the image at the center"""

    def __init__(self):
        self._init(locals())

    def _augment(self, img, _):
        orig_shape = img.shape
        crop_shape = min(orig_shape[0], orig_shape[1])
        h0 = int((orig_shape[0] - crop_shape) * 0.5)
        w0 = int((orig_shape[1] - crop_shape) * 0.5)
        return img[h0:h0 + crop_shape, w0:w0 + crop_shape]

    def _fprop_coord(self, coord, param):
        raise NotImplementedError()

class ImageOutputsFromFile(RNGDataFlow):
    """ Produce images read from a list of files. Adapted version of ImageFromFile in tensorpack"""
    def __init__(self, files_outputs, channel=3, resize=None, shuffle=False):
        """
        Args:
            files (list): list of touples with file path and outputs.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files_outputs), "No image files given to ImageFromFile!"
        self.files_outputs = files_outputs
        self.channel = int(channel)
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        if resize is not None:
            resize = shape2d(resize)
        self.resize = resize
        self.shuffle = shuffle


    def size(self):
        return len(self.files_outputs)


    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.files_outputs)
        for f in self.files_outputs:
            im = cv2.imread(f[0], self.imread_mode)
            if self.channel == 3:
                im = im[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                im = im[:, :, np.newaxis]
            yield [im, f[1], np.array(f[0])]


class Model(ModelDesc):

    def __init__(self, arch):
        super(Model, self).__init__()
        self.arch = arch

    def _get_inputs(self):
        """Define intputs to network. tensorpack requires this function"""
        if self.arch == 'orig':
            return [InputDesc(tf.float32, [None, 472, 472, 3], 'input'),
                    InputDesc(tf.float32, [None, 2], 'cords'),
                    InputDesc(tf.string, [None], 'img_path'),
                    ]
        else if 'paper' in self.arch:
            return [InputDesc(tf.float32, [None, 256, 256, 3], 'input'),
                    InputDesc(tf.float32, [None, 2], 'cords'),
                    InputDesc(tf.string, [None], 'img_path'),
                    ]

    def _build_graph(self, inputs):
        """Define graph = network. tensorpack requires this function"""
        image, cords, img_path = inputs
        is_training = get_current_tower_context().is_training
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        if is_training:
            tf.summary.image("train_image", image, 10)

        image = image / 4.0 # make range smaller after 
        
        hn = [64, 64, 128, 128, 128, 128, 256, 90]
        if self.arch == 'orig':
            with argscope(Conv2D, nl=tf.nn.relu, use_bias=False, kernel_shape=3):

                # Block 1: 256x256x3 --> 128x128x(hn[0])
                x = Conv2D('conv1_1', image, hn[0])
                x = Conv2D('conv1_2', x, hn[0])
                x = MaxPooling('pool1', x, 3, stride=2, padding='SAME')

                # Block 2: 128x128x(hn[0]) --> 64x64x(hn[1])
                x = Conv2D('conv2_1', x, hn[1])
                x = Conv2D('conv2_2', x, hn[1])
                x = MaxPooling('pool2', x, 3, stride=2, padding='SAME')

                # Block 3: 32x32x(hn[1]) --> 16x16x(hn[2])
                x = Conv2D('conv3_1', x, hn[2])
                x = Conv2D('conv3_2', x, hn[2])
                x = MaxPooling('pool3', x, 3, stride=2, padding='SAME')

                # Block 4: 16x4x(hn[2]) --> 2x8x(hn[3])
                x = Conv2D('conv4_1', x, hn[3])
                x = Conv2D('conv4_2', x, hn[3])
                x = MaxPooling('pool4', x, 3, stride=2, padding='SAME')

                # Block 4: 16x4x(hn[3]) --> 2x4x(hn[4])
                x = Conv2D('conv5_1', x, hn[4])
                x = Conv2D('conv5_2', x, hn[4])
                x = MaxPooling('pool5', x, 3, stride=2, padding='SAME')

                # Block 4: 16x4x(hn[4]) --> 2x2x(hn[5])
                x = Conv2D('conv6_1', x, hn[5])
                x = Conv2D('conv6_2', x, hn[5])
                x = MaxPooling('pool6', x, 3, stride=2, padding='SAME')

                # 2x2x(hn[5]) --> (hn[6])
                x = FullyConnected('fc7', x, hn[6], nl=tf.nn.relu)

                # (hn[6]) --> (hn[7])
                x = FullyConnected('fc8', x, (hn[7]), nl=tf.nn.relu)
                x = tf.identity(x, name='fc_activation')  # to allow for extraction of activations

                y_out = FullyConnected('fc9', x, out_dim=2, nl=tf.identity)
        elif 'paper' in self.arch:
              with argscope(Conv2D, nl=BNReLU):
                # 472x472x3 --> 236x236x64
                x = Conv2D('conv1_1', image, hn[0], 6, stride=2)
                # 236x236x64 --> 78x78x64
                x = MaxPooling('pool1', x, 3)

                # 78x78x64 --> 78x78x64
                x = Conv2D('conv2', x, hn[1], 5)
                x = Conv2D('conv3', x, hn[1], 5)
                x = Conv2D('conv4', x, hn[1], 5)
                x = Conv2D('conv5', x, hn[1], 5)
                x = Conv2D('conv6', x, hn[1], 5)
                x = Conv2D('conv7', x, hn[1], 5)

                # 78x78x64 --> 26x26x64
                x = MaxPooling('pool2', x, 3)

                if self.arch == 'paper_ext':
                    # 26x26x64 --> 26x26x64
                    x = Conv2D('conv8', x, hn[1], 3)
                    x = Conv2D('conv9', x, hn[1], 3)
                    x = Conv2D('conv10', x, hn[1], 3)
                    x = Conv2D('conv11', x, hn[1], 3)
                    x = Conv2D('conv12', x, hn[1], 3)
                    x = Conv2D('conv13', x, hn[1], 3)
                    # 26x26x64 --> 13x13x64
                    x = MaxPooling('pool3', x, 2)
                    x = Conv2D('conv14', x, hn[1], 3)
                    x = Conv2D('conv15', x, hn[1], 3)
                    x = Conv2D('conv16', x, hn[1], 3)

                # 13x13x64 --> 64
                x = FullyConnected('fc17', x, 64, nl=tf.nn.relu)

                # (64) --> (64)
                x = FullyConnected('fc18', x, 64, nl=tf.nn.relu)
                x = tf.identity(x, name='fc_activation')    # to allow for extraction of activations

                y_out = FullyConnected('fc19', x, out_dim=2, nl=tf.identity)

        y_ = tf.identity(y_out, name='out_activation')    # to allow for extraction of outputs
        cost = tf.reduce_mean(tf.square(y_ - cords), name='mse_cost')

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(1e-4), name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 1e-3, summary=True)
        return tf.train.AdamOptimizer(lr, epsilon=1e-3)


def get_data(train_or_test, batch_size=32, scale=True):
    """Datastream for images for CNN regression.
    Inputs:
            traindata_or_testdata: whether to use train or test data. Valid values: 'train' or 'test'
    Returns:
            Datastream
    """
    augs = [CenterCropMax()]
    if PAPER:
        augs.append(imgaug.Resize((472, 472)))
    else:
        augs.append(imgaug.Resize((256, 256)))

    if scale:
        augs.append(Scale(output_domain=[0., 1.]))

    # Get paths to train/test images and their true coordinates
    import csv
    imgs_outputs = []
    with open('{}_list.csv'.format(train_or_test), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img_path = 'pics2/pic_{}.jpeg'.format(row[0])
            x1, x2 = row[1].strip('(').strip(')').split(',')
            imgs_outputs.append((img_path, np.array((float(x1), float(x2)))))

    ds = ImageOutputsFromFile(imgs_outputs, channel=3, shuffle=True)
    ds = PrefetchData(ds, 4) # use queue size 4
    ds = AugmentImageComponent(ds, augs)
    ds = BatchData(ds, batch_size)

    # Plot examples of transformed images to see how they look:
    if DEBUG:
        batch = next(ds.get_data())
        f, ax = plt.subplots(2, 2)
        ax = ax.flatten()
        map(lambda i: ax[i].imshow(batch[0][i]), range(4))
        f.savefig('./check_imgs.jpeg')
    return ds

def train(log_dir, load=None, gpu=None, arch='orig'):
    with tf.Graph().as_default():
        config = get_config(log_dir=log_dir, arch='orig')
        if load is not None:
            config.session_init = SaverRestore(load)
        if gpu is not None:
            config.nr_tower = len(gpu.split(','))
        nr_gpu = get_nr_gpu() 
        if nr_gpu == 1:
            QueueInputTrainer(config).train()
        else:
            SyncMultiGPUTrainer(config).train()


def get_config(log_dir=None, arch='orig'):
    if log_dir is None:
        logger.auto_set_dir(action='k')
    else:
        logger.set_logger_dir(log_dir, action='k')

    # prepare dataset
    dataset_train = get_data('train')
    dataset_test = get_data('test')

    def lr_func(lr):
        if lr < 1e-5:
            raise StopTraining()
        return lr * 0.31
    return TrainConfig(
        model=Model(arch),
        dataflow=dataset_train,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=10),
            InferenceRunner(dataset_test, ScalarStats('mse_cost')),
            StatMonitorParamSetter('learning_rate', 'validation_mse_cost', lr_func,
                                   threshold=1e-16, last_k=50),
        ],
        max_epoch=1000,
    )


def network_prediction(imgs, model_path):
    """Returns feature activations of the last fully connected layer. Assumes that the images are already
    preprocessed as in the training/test case (using get_data).
    """

    pred = OfflinePredictor(
        PredictConfig(
            session_init=get_model_loader(model_path),
            model=Model(),
            input_names=['input'],
            output_names=['out_activation']
        )
    )

    prediction = np.asarray(pred([imgs])[0])

    return prediction


def get_predicitions(model_base_dir, train_or_test):
    df = get_data(train_or_test)
    df.reset_state()
    ds = df.get_data()
    dp = next(ds)

    model_file = get_checkpoint_file_cifar(model_base_dir)
    preds = network_prediction(dp[0], model_file)
    return dp, preds

def get_checkpoint_file_cifar(checkpoint_dir):
    """Get checkpoint file_name"""
    import glob
    import logging
    checkpoint_file = glob.glob(os.path.join(checkpoint_dir, 'model*.index'))
    if not checkpoint_file:
        logging.warning("No checkpoint file found. Check utils.get_checkpoint_file()")
    else:
        final_cf = get_cf_with_highest_iter(checkpoint_file)
        logging.info("Using checkpoint file: " + final_cf)

    return final_cf.rsplit('.index')[0]

def get_cf_with_highest_iter(checkpoint_files):
    """ Get the checkpoint file with highest iteration number."""
    highest_num = -1
    final_cf = None
    for cf in checkpoint_files:
        print cf
        import re
        m = re.search(r'model-([0-9]+)\.index', cf)
        if m:
            curr_num = int(m.group(1))
            highest_num = max(curr_num, highest_num)
            if highest_num == curr_num:
                final_cf = cf
    return final_cf

def plot_predicitions(dp, preds, save_path):
    """ dp - list of datapoint with: 
        [image [batch_size, n_dim, n_dim, 3], 
         coordinates [x1, x2],
         path [img_path]
        ]
        preds - list of predicition for data points, aligned by index
        [coordinates [x1, x2]
        ]
    """
    f, ax = plt.subplots(1, 1)
    for i in range(len(dp[1])):
        import re
        m = re.search('_([0-9]+_[0-9]+).', dp[2][i]) # extract image id from the path
        if m:
            img_id = m.group(1)
        else:
            img_id = 'UNK'
        x1_true, x2_true = dp[1][i,0], dp[1][i,1]
        x1_pred, x2_pred = preds[i,0], preds[i,1]
        ax.scatter(x1_true, x2_true, c='b')
        ax.text(x1_true, x2_true, img_id, size='8')
        ax.scatter(x1_pred, x2_pred, c='r')
        ax.text(x1_pred, x2_pred, img_id, size = '8')
    f.savefig(save_path)

def plot_preds_on_imgs(dp, preds, save_path):
    """ dp - list of datapoint with: 
        [image [batch_size, n_dim, n_dim, 3], 
         coordinates [x1, x2],
         path [img_path]
        ]
        preds - list of predicition for data points, aligned by index
        [coordinates [x1, x2]
        ]
    """
    n_dim = dp[0][0].shape[0] # take dimension from the first images (the same for all images)

    def reorder_and_scale_cords(cords, new_scale):
        """ Scales the coordinates to image dimension size 
        and reorder to be positively correlated with red gear coordinates.
        """
        cords *= new_scale # scale 
        tmp = np.copy(cords) # swap dimensions
        cords[:,0] = n_dim - tmp[:,1] - 50
        cords[:,1] = n_dim - tmp[:,0]
        return cords

    dp[1] = reorder_and_scale_cords(dp[1], n_dim)
    preds = reorder_and_scale_cords(preds, n_dim)

    f, ax = plt.subplots(5, 5)
    ax = ax.flatten()
    for i in range(len(ax)):
        m = re.search('_([0-9]+_[0-9]+).', dp[2][i])
        if m:
            img_id = m.group(1)
        else:
            img_id = 'UNK'
        x1_true, x2_true = dp[1][i,0], dp[1][i,1]
        x1_pred, x2_pred = preds[i,0], preds[i,1]
        img = dp[0][i]
        ax[i].set_title(img_id, fontsize=5)
        ax[i].scatter(x1_true, x2_true, c='g', marker='x', s=8)
        ax[i].scatter(x1_pred, x2_pred, c='y', marker='d', s=5)
        ax[i].imshow(img)
        ax[i].axis('off')
    pred_patch = mpatches.Patch(color='y', label='Model prediction')
    true_patch = mpatches.Patch(color='g', label='True coordinates ')
    plt.legend(handles=[true_patch, pred_patch])
    f.savefig(save_path, dpi=400)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', default=None)
    parser.add_argument('--mode', help='action to perform. ', default=None  )
    parser.add_argument('--load', help='load model', default=None)
    parser.add_argument('--logdir', help='where to log results?', default='results')
    parser.add_argument('--arch', help='type of network architecture to use (options: orig, paper, paper_ext)', default='orig')
    # Parameters required by Neptune
    parser.add_argument('--job-id', help='neptune arg', default=None)
    parser.add_argument('--rest-api-url', help='neptune arg', default=None)
    parser.add_argument('--ws-api-url', help='neptune arg', default=None)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.gpu is None: # If loading through Neptune
        args.mode = context.params.mode
        args.gpu = context.params.gpu
        args.logdir = context.params.logdir
        args.arch = context.params.arch

    if args.mode == 'train':
        train(args.logdir, args.load, args.gpu, args.arch)

    elif args.mode == 'predict':
        dp, preds = get_predicitions(args.load, 'test', args.arch)
        plot_predicitions(dp, preds, 'predictions.jpeg')
        plot_preds_on_imgs(dp, preds, 'preds_on_imgs.jpeg')