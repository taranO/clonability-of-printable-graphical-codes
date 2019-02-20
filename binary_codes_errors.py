'''
Hamming distance between the binary codes
'''
import argparse
import matplotlib.image as mpimg

from github.libs.utils import *

# ----------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--code', type=str, default='sa', help='Printer type: sa, lx, hp or ca')
parser.add_argument('--model_type', type=str, default='fc', help='bn: bottelneck model, fc: fully connected')
parser.add_argument('--model_size', type=str, default='3', help='number of hidden layers in fc model. For bn it should be empty')

args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    DIR_ORG = "data/codes_%s/" % args.code
    DIR_REG = "data/regenerated/%s%s/codes_%s/" % (args.model_type, args.model_size, args.code)

    # --------------------------------------------------------------------------------
    list = os.listdir(DIR_ORG)
    list.sort()
    N = len(list)

    ERR = []
    for ind in range(0,384):
        img_org = np.asarray(mpimg.imread(DIR_ORG + list[ind])).astype(np.float64)/255
        img_reg = np.asarray(mpimg.imread(DIR_REG + list[ind])).astype(np.float64)/255

        ERR.append(100*np.mean(np.logical_xor(img_org, img_reg)))

    ERR = np.asarray(ERR)
    print("code=%s, mean of xor error = %0.4f" % (args.code, np.mean(ERR)))











