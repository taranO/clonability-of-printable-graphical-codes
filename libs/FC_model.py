from torch import nn
import torch.nn.init as init

class FCRegression_2layers(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(FCRegression_2layers, self).__init__()

        self.fc = nn.Sequential(
            # 1 layer
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            # 2 layer
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(True),
            # output
            nn.Linear(n_hidden, n_input)
        )

        self.relu = nn.ReLU(True)

        self.fc.apply(self.init_weights)

    def forward(self, x):
        return self.fc(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            #init.xavier_uniform(m.weight)
            init.xavier_normal(m.weight)

class FCRegression_3layers(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(FCRegression_3layers, self).__init__()

        self.fc = nn.Sequential(
            # 1 layer
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            # 2 layer
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(True),
            # 3 layer
            nn.Linear(n_hidden, n_input),
            #nn.BatchNorm1d(n_input),
            nn.Tanh(),
        )

        self.relu = nn.ReLU(True)

        self.fc.apply(self.init_weights)

    def forward(self, x):
        return self.fc(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            #init.xavier_uniform(m.weight)
            init.xavier_normal(m.weight)

class FCRegression_4layers(nn.Module):
    def __init__(self, n_input, n_hidden):
        super(FCRegression_4layers, self).__init__()

        self.fc = nn.Sequential(
            # 1 layer
            nn.Linear(n_input, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(True),
            # 2 layer
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.Tanh(),
            # 3 layer
            nn.Linear(n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(True),
            # 4 layer
            nn.Linear(n_hidden, n_input),
            #nn.BatchNorm1d(n_input),
            nn.Tanh(),
        )

        self.relu = nn.ReLU(True)

        self.fc.apply(self.init_weights)

    def forward(self, x):
        return self.fc(x)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            #init.xavier_uniform(m.weight)
            init.xavier_normal(m.weight)
