'''
Estimate an optimal threshold on validation set
'''

import argparse
import numpy as np
from torch.autograd import Variable

from github.libs.BN_model import *
from github.libs.FC_model import *
from github.libs.utils import *

# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--code', type=str, default='sa', help='Printer type: sa, lx, hp or ca')
parser.add_argument('--model_type', type=str, default='fc', help='bn: bottelneck model, fc: fully connected')
parser.add_argument('--model_size', type=str, default='3', help='number of hidden layers in fc model. For bn it should be empty')
parser.add_argument('--input_size', type=int, default=24, help='input size')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train')

args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    h = 384 # full image size
    w = 384 # full image size

    # model parameters
    n_input = args.input_size ** 2
    if args.model_type == "bn": # bottelneck model parameters
        n_hidden = [256, 128]
        n_output = 36
    elif args.model_type == "fc": # fully connected model parameters
        n_hidden = n_input

    MODEL_DIR = "models/%s%s_codes_%s/" % (args.model_type, args.model_size, args.code)

    # --------------------------------------------------------------------------------
    # load data
    original = np.load("data/codes_original_%dx%d_validation.npy" % (args.input_size, args.input_size))
    data_to_test = np.load("data/codes_%s_%dx%d_validation.npy" % (args.code, args.input_size, args.input_size))

    original = prepareData(original, is_model_linear=True)
    data_to_test = prepareData(data_to_test, is_model_linear=True)

    # --------------------------------------------------------------------------------
    # model
    if args.model_type == "bn":
        model = BNModel(n_input, n_hidden, n_output)
    elif args.model_type == "fc" and args.model_size == "2":
        model = FCRegression_2layers(n_input, n_hidden)
    elif args.model_type == "fc" and args.model_size == "3":
        model = FCRegression_3layers(n_input, n_hidden)
    elif args.model_type == "fc" and args.model_size == "4":
        model = FCRegression_4layers(n_input, n_hidden)

    model = toDevice(model)

    # --------------------------------------------------------------------------------

    Thr = np.arange(0, 1, 0.05)
    Res = {}
    MN = []
    for epoch in range(args.n_epochs):
        model.load_state_dict(
            torch.load(MODEL_DIR + "/epoch_%d.pth" % epoch, map_location=lambda storage, loc: storage))

        Dist = []
        DistJS = {}
        for thr in Thr:
            org = []
            rec = []
            sc = []

            for i in range(0, data_to_test.shape[0], 256):
                input = toDevice(Variable(torch.Tensor(data_to_test[i:i + 256])))
                output = model(input)
                output[output < thr] = 0
                output[output >= thr] = 1

                # reconstruction to full image
                rec_set = output.cpu().data.numpy()
                rec_img = BlocksToImage(rec_set, args.input_size, args.input_size, h, w)
                rec.append(np.reshape(rec_img, (h * w)))

                # original to full image
                org_img = BlocksToImage(original[i:i + 256], args.input_size, args.input_size, h, w)
                org.append(np.reshape(org_img, (h * w)))

            rec = np.asarray(rec)
            org = np.asarray(org)

            dist = np.power((org - rec), 2)
            dist = np.sum(dist, 1)
            dist = dist / org.shape[1]
            Dist.append(np.mean(dist))

            DistJS.update({thr: np.mean(dist)})

            print("epoch=%d, thr=%0.4f mse=%0.6f" % (epoch, thr, np.mean(dist)))

        res = np.min(Dist)
        thr = np.argmin(Dist)

        MN.append(res)
        print("\n\t epoch=%d, thr=%0.4f mse=%0.6f\n" % (epoch, Thr[thr], res))

        Res.update({epoch: DistJS})

    print("\n min = %0.6f" % np.min(np.asarray(MN)))


