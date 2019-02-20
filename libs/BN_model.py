from torch import nn
import torch.nn.init as init

class BNModel(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(BNModel, self).__init__()

        self.encoder = nn.Sequential(
            # 1 layer
            nn.Linear(n_input, n_hidden[0]),
            nn.BatchNorm1d(n_hidden[0]),
            nn.Tanh(),
            # 2 layer
            nn.Linear(n_hidden[0], n_hidden[1]),
            nn.BatchNorm1d(n_hidden[1]),
            nn.ReLU(True),
            # output
            nn.Linear(n_hidden[1], n_output)
        )
        self.decoder = nn.Sequential(
            # 1 layer
            nn.Linear(n_output, n_hidden[1]),
            nn.BatchNorm1d(n_hidden[1]),
            nn.Tanh(),
            # 2 layer
            nn.Linear(n_hidden[1], n_hidden[0]),
            nn.ReLU(True),
            # output
            nn.Linear(n_hidden[0], n_input),
            nn.ReLU(True),
        )

        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            #init.xavier_uniform(m.weight)
            init.xavier_normal(m.weight)


