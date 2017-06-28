import argparse
import logging
import tensorflow as tf
import tensorpack as tp
import numpy as np

from classify_image import *
from cnn_regression import *

def extract_features(file_name, train_or_test):
    df = get_data(train_or_test, batch_size=1, scale=False)
    df.reset_state()
    ds = df.get_data()
    
    model_dir = 'model/'
    maybe_download_and_extract(model_dir)
    create_graph(model_dir)
    with tf.Session() as sess:
        cnn_codes_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        cnn_codes = np.zeros((df.size(), 2048), dtype='float32')
        orig_imgs = np.zeros((df.size(), 256, 256, 3), dtype='float32')
        cnt = 0
        y = []
        paths = []
        for d in ds:
            codes = sess.run([cnn_codes_tensor], {'DecodeJpeg:0': np.squeeze(d[0])})
            if (cnt % 10 == 0):
                logging.info("{}/{} images completed.".format(cnt, df.size()))
            orig_imgs[cnt] = d[0]
            cnn_codes[cnt] = np.squeeze(codes)
            cnt += 1
            y.append(np.squeeze(d[1])) # Saving to store together with the codes
            paths.append(np.squeeze(d[2])) # Saving to store together with the codes
        np.savez_compressed(file_name + '_' + train_or_test + ".npz", cnn_codes=cnn_codes, orig_imgs=orig_imgs, y=y, paths=paths)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    modes = ['train', 'test']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=modes[0], choices=modes,
                      help='Train or test dataset. Possible values: {}'.format(', '.join(modes)))
    parser.add_argument('--savepath', type=str, default='cnncodes',
                      help='Path where to store the extracted features.')
    args = parser.parse_args()

    extract_features(args.savepath, args.mode)
