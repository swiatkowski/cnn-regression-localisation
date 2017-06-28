import argparse
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale

import scipy

from utils import *

def plot_tsne_cnncodes(save_path, max_examples = 1000):
    """ 
    X_test: [n_examples, 2048]
    y_test: [n_exampels, 2]
    orig_test: [n_examples, n_orig_dim, n_orig_dim, 3]
    """
    logging.info('Loading CNN codes...')
    data_test = np.load('cnncodes_test.npz')
    X_test, y_test, orig_test = data_test['cnn_codes'], data_test['y'], data_test['orig_imgs']
    n_classes = len(np.unique(y_test))
    n_examples = min(max_examples, y_test.shape[0])
    n_orig_dim = orig_test.shape[1]
    X_test = X_test[:n_examples,:]
    
    X_test_tsne = TSNE(n_components=2, random_state=0, n_iter=1000, verbose=1).fit_transform(X_test)
    
    plt.scatter(X_test_tsne[:,0], X_test_tsne[:,1], s=10, edgecolors='none')
    plt.axis('off')
    plt.savefig('{}_tsne_scatter.png'.format(save_path), bbox_inches='tight')

    # Plot the original images in the t-SNE dimensions
    RES = 2000
    img = np.zeros((RES,RES,3),dtype='uint8')
    X_test_tsne  = minmax_scale(X_test_tsne)
    for i in range(n_examples):
        x1_scaled = int(X_test_tsne[i,0] * (RES - n_orig_dim))
        x2_scaled = int(X_test_tsne[i,1] * (RES - n_orig_dim))
        img[x1_scaled:x1_scaled+n_orig_dim,x2_scaled:x2_scaled+n_orig_dim,:] = orig_test[i]
    plt.imshow(img)

    scipy.misc.imsave('{}_tsne_orig_imgs.jpg'.format(save_path), img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    modes = ['plot_cnn']
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default=modes[0], choices=modes,
                      help='Operation to perform. Possible values: {}'.format(', '.join(modes)))
    args = parser.parse_args()

    if args.mode == 'plot_cnn': # Plots t-SNE embeddings of the cnn codes
        plot_tsne_cnncodes('cnn')
    else:
        logging.warning('Uknown mode. Possible values: {}'.format(', '.join(modes)))
