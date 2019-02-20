import os
import argparse
import numpy as np
import matplotlib.image as mpimg

import github.libs.utils as utils

# ================================================================================================
# functions

def loadData(dir, n_train=100, n_valid=50, n_test=234):
    Train = []
    Valid = []
    Test  = []

    list = os.listdir(dir)
    list.sort()
    N = len(list)

    for ind in range(N):
        print(list[ind])

        img = np.asarray(mpimg.imread(dir + list[ind])).astype(np.float64)
        Blocks = np.asarray(utils.toBlocks(img, args.input_size, is_non_overlap=True))

        # Blocks -= np.min(Blocks, axis=(1, 2), keepdims=True)
        mx = np.max(Blocks, axis=(1, 2), keepdims=True)
        mx[mx == 0] = 1
        Blocks /= mx

        if ind < n_train:
            Train.append(Blocks)
        elif ind < n_train+n_valid:
            Valid.append(Blocks)
        else:
            Test.append(Blocks)

    return np.array(Train), np.array(Valid), np.array(Test)

# ================================================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--input_size', type=int, default=24, help='input size')
parser.add_argument('--n_train', type=int, default=100, help='number of samples for train')
parser.add_argument('--n_valid', type=int, default=50, help='number of samples for validation')
parser.add_argument('--n_test', type=int, default=234, help='number of samples for test')
parser.add_argument('--code', type=str, default='sa', help='Printer type: sa, lx, hp, ca or original')

args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    data_dir = "data/codes_%s/" % args.code

    train, validation, test = loadData(data_dir, args.n_train, args.n_valid, args.n_test)

    train      = np.reshape(train, (-1, args.input_size**2))
    validation = np.reshape(validation, (-1, args.input_size**2))
    test       = np.reshape(test, (-1, args.input_size**2))

    np.save("data/codes_%s_%dx%d_train.npy" % (args.code, args.input_size, args.input_size), train)
    np.save("data/codes_%s_%dx%d_validation.npy" % (args.code, args.input_size, args.input_size), validation)
    np.save("data/codes_%s_%dx%d_test.npy" % (args.code, args.input_size, args.input_size), test)