import os
import torch
import numpy as np

def toDevice(Obj):
    if torch.cuda.is_available():
        Obj = Obj.cuda()

    return Obj


def toBlocks(img, b, is_non_overlap=False, step=1):
    B = []

    if is_non_overlap:
        h, w = img.shape
        for r in range(0, h, b):
            for c in range(0, w, b):
                blk = img[r:r + b, c:c + b]
                if blk.shape[0] != b or blk.shape[1] != b:
                    continue
                B.append(blk)
    elif step == 1:
        h, w = img.shape
        for r in range(h):
            for c in range(w):
                blk = img[r:r + b, c:c + b]
                if blk.shape[0] != b or blk.shape[1] != b:
                    continue
                B.append(blk)
    else:
        h, w = img.shape
        for r in np.arange(0, h, step):
            for c in np.arange(0, w, step):
                if r + b > h:
                    r = h - b
                if c + b > w:
                    c = w - b

                blk = img[r:r + b, c:c + b]

                B.append(blk)

    return B

def prepareData(data, input_size=24, is_model_linear=False):
    if is_model_linear:
        data = np.reshape(data, (-1, input_size**2))
    else:
        data = np.reshape(data, (-1, 1, input_size, input_size))

    return data

def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

'''
    Collects blocks to image
'''
def BlocksToImage(blocks, b, stride,  heigh, width):
    iI = np.zeros((heigh, width))
    count = np.zeros((heigh, width))

    i = 0
    for r in range(0, heigh, stride):
        for c in range(0, width, stride):
            blk = np.reshape(blocks[i], (b, b))
            iI[r:r+b, c:c+b] = iI[r:r+b, c:c+b] + blk
            count[r:r+b, c:c+b] += 1
            i += 1

    iI = np.divide(iI, count)

    return iI