'''
Generate fake codes from scanned originals
'''
import argparse
import scipy.misc
import matplotlib.image as mpimg

from torch.autograd import Variable

from github.libs.BN_model import *
from github.libs.FC_model import *
from github.libs.utils import *

# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=9, help='epoch to test')
parser.add_argument('--thr', type=float, default=0.5, help='estimated optimal threshold')
parser.add_argument('--input_size', type=int, default=24, help='input size')
parser.add_argument('--code', type=str, default='sa', help='Printer type: sa, lx, hp or ca')
parser.add_argument('--model_type', type=str, default='fc', help='bn: bottelneck model, fc: fully connected')
parser.add_argument('--model_size', type=str, default='3', help='number of hidden layers in fc model. For bn it should be empty')

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

    MODEL_DIR = "models/%s%s_codes_%s" % (args.model_type, args.model_size, args.code)

    DATA_DIR  = "data/codes_%s/" % args.code
    SAVE_TO   = "data/regenerated/%s%s/codes_%s/" % (args.model_type, args.model_size, args.code)
    makeDir(SAVE_TO)

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

    model.load_state_dict(torch.load(MODEL_DIR + "/epoch_%d.pth" % args.epoch, map_location=lambda storage, loc: storage))

    # --- start regeneration -----------------------
    list = os.listdir(DATA_DIR)
    list.sort()
    N = len(list)

    for ind in range(N):

        img = np.asarray(mpimg.imread(DATA_DIR + list[ind])).astype(np.float64)
        Data = np.asarray(toBlocks(img, args.input_size, is_non_overlap=True))

        # normalization: zero mean & unit norm
        Data -= np.min(Data, axis=(1, 2), keepdims=True)
        mx = np.max(Data, axis=(1, 2), keepdims=True)
        mx[mx == 0] = 1
        Data /= mx
        Data = np.reshape(Data, (-1, args.input_size ** 2))

        output = model(toDevice(Variable(torch.Tensor(Data))))
        output[output < args.thr] = 0
        output[output >= args.thr] = 1

        rec_img = BlocksToImage(output.cpu().data.numpy(), args.input_size, args.input_size, h, w)

        scipy.misc.imsave(SAVE_TO + list[ind], rec_img)


