'''
Model training
'''


import argparse
from torch.autograd import Variable

from github.libs.BN_model import *
from github.libs.FC_model import *
from github.libs.utils import *

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default='fc', help='bn: bottelneck model, fc: fully connected')
parser.add_argument('--model_size', type=str, default='4', help='number of hidden layers in fc model. For bn it should be empty')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for optimazer')
parser.add_argument('--wd', type=float, default=1e-5, help='weight decay for optimazer')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--input_size', type=int, default=24, help='input size')
parser.add_argument('--code', type=str, default='sa', help='Printer type: sa, lx, hp or ca')
parser.add_argument('--optim', type=str, default='Adam', help='optimization default:SGD, option: Adam')
parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--save_each', type=int, default=1, help='save each X epochs')

args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # model parameters
    n_input = args.input_size ** 2
    if args.model_type == "bn": # bottelneck model parameters
        n_hidden = [256, 128]
        n_output = 36
    elif args.model_type == "fc": # fully connected model parameters
        n_hidden = n_input

    # ------------------------------------------------------------------------------------------------------------------
    # prepare training data
    original = np.load("data/codes_original_%dx%d_train.npy" % (args.input_size, args.input_size))
    scanned  = np.load("data/codes_%s_%dx%d_train.npy" % (args.code, args.input_size, args.input_size))

    n_samples = scanned.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    if args.model_type == "bn":
        model = BNModel(n_input, n_hidden, n_output)
    elif args.model_type == "fc" and args.model_size == "2":
        model = FCRegression_2layers(n_input, n_hidden)
    elif args.model_type == "fc" and args.model_size == "3":
        model = FCRegression_3layers(n_input, n_hidden)
    elif args.model_type == "fc" and args.model_size == "4":
        model = FCRegression_4layers(n_input, n_hidden,)
    model = toDevice(model)

    criterion = toDevice(nn.MSELoss())
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        permutation = np.random.permutation(n_samples)
        for i in range(0, scanned.shape[0], args.batch_size):
            indices   = permutation[i:i+args.batch_size]
            batch_x   = toDevice(Variable(torch.Tensor(scanned[indices])))
            batch_org = toDevice(Variable(torch.Tensor(original[indices])))

            # ===================forward=====================
            output = model(batch_x)
            loss = criterion(output, batch_org)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================log========================
        print("epoch [{}/{}], loss:{:.10f}" . format(epoch + 1, args.n_epochs, loss.data[0]))

        if epoch % args.save_each == 0 or epoch == args.n_epochs:
            torch.save(model.state_dict(),
                       "models/%s%s_codes_%s/epoch_%d.pth" % (args.model_type, args.model_size, args.code, epoch))



