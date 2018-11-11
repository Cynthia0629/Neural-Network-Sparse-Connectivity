import numpy as np
# torch
import torch
import torch.multiprocessing as mp
from torch import optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
# dev
from dev.utils import data_tools
from dev.models.gmm_lrt import GmmLrt
import dev.utils.data_tools as data_tools
from dev.models.neural_nets import *

import dev.models.custom_losses as cl

def train_network(model, data, labels, val_data=None, val_labels=None, lr=0.5,
                  batch_size=32, optimizer='SGD', max_iter=1000, min_iter=0,
                  tol=1e-4, verbose=10):
    """Train a neural network
    """

    # Check for cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Running on GPU')

    # Define the architecture of the DNN
    if len(data.size()) == 2:
        n, nfeats = data.size()
    if len(data.size()) == 3:
        n, nchns, nfeats = data.size()
    nbatches = np.ceil(n / batch_size)

    # Torchify the data
    model.train(mode=True)
    if use_cuda:
        data = data.cuda()
        labels = labels.cuda()
        model = model.cuda()
    model.train_history = []
    model.has_history = True

    # If val_data is present, torchify
    if val_data is not None:
        if use_cuda:
            val_data = data.cuda()
            val_labels = labels.cuda()
        model.has_val_history = True
        model.val_history = []

    # Loop until convergence
    curr_iter = 0
    old_loss = np.inf
    done = False
    opt_str = "optim.{}(model.parameters(), lr = {})".format(optimizer,
                                                             lr / nbatches)
    opt = eval(opt_str)
    while not done:
        # Get minibaches and loop
        batch_indices = data_tools.get_batches(n, batch_size)
        loss = 0
        for idx in batch_indices:
            batch_loss = model.compute_loss(data[idx, :], labels[idx, ])

            # Backprop and 1-2 step
            opt.zero_grad()
            model.zero_grad()
            batch_loss.backward()
            opt.step()

        # Compute the loss for the whole data
        model.eval()
        if use_cuda:
            curr_loss = model.compute_loss(data, labels).cpu().data.numpy()
        else:
            curr_loss = model.compute_loss(data, labels).data.numpy()
        model.train_history.append(curr_loss)
        model.train(mode=True)

        # If there is validation data, compute the val loss
        if val_data is not None:
            model.eval()
            if use_cuda:
                val_loss = model.compute_loss(val_data,
                                              val_labels).cpu().data.numpy()
            else:
                val_loss = model.compute_loss(val_data,
                                              val_labels).data.numpy()
            model.val_history.append(val_loss)
            model.train(mode=True)

        scaled_diff = 20
        if curr_iter > 0:
            scaled_diff = (abs(curr_loss - old_loss) / np.abs(old_loss))
        old_loss = curr_loss

        if verbose > 0 and curr_iter % verbose == 0:
            loss_str = 'Iter {}: Loss: {} dL/dt: {}'.format(curr_iter,
                                                            curr_loss,
                                                            scaled_diff)
            print(loss_str)

        if curr_iter >= min_iter:
            if scaled_diff < tol:
                done = True
                if verbose > 0:
                    print('Reached convergence')
            if curr_iter >= max_iter:
                done = True
                if verbose > 0:
                    print('Reached maximum iterations')
        curr_iter += 1

    # Put the model in evaluation mode
    model.eval()
    if use_cuda:
        model = model.cpu()
    return model


class Dnn(nn.Module):
    def __init__(self, nfeats, layers, p_dropout=0.0, class_weight=None):
        super(Dnn, self).__init__()
        
        self.p_dropout = p_dropout
        self.layers = layers
        self.class_weight = class_weight

        self.has_history = False
        self.has_val_history = False

        self.fc_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Add first layer
        self.nlayers = len(layers)
        # Add first layer
        self.fc_layers.append(nn.Linear(nfeats, layers[0]))
        self.dropouts.append(nn.Dropout(p=p_dropout))
        self.activations.append(nn.ReLU())
        # Add further hidden layers
        for ii in range(1, self.nlayers):
            self.fc_layers.append(nn.Linear(layers[ii-1], layers[ii]))
            self.dropouts.append(nn.Dropout(p=p_dropout))
            self.activations.append(nn.ReLU())
        # Add output layer
        self.output_layer = nn.Linear(layers[self.nlayers-1], 2)
        self.softmax = nn.Softmax(1)
        self.loss_function = torch.nn.CrossEntropyLoss()
        print('sk')
        # self.loss_function = cl.CELoss()

    def encode(self, x):
        # First layer 
        h = self.dropouts[0](self.activations[0](self.fc_layers[0](x)))
        # Keep going
        for ii in range(1, self.nlayers):
            h = self.dropouts[ii](self.activations[ii](self.fc_layers[ii](h)))
        return h

    def forward(self, x):
        h = self.encode(x)
        output = self.output_layer(h)
        return output

    def predict_proba(self, x):
        x = torch.from_numpy(x)
        x = Variable(x.float())
        return self.softmax(self.forward(x)).data.numpy()

    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        loss = self.loss_function(y_pred, y)
        return loss


# class UncertainDnn(nn.Module):
#     def __init__(self, nfeats, layers, p_dropout=0.0):
#         super(UncertainDnn, self).__init__()
        
#         self.p_dropout = p_dropout
#         self.layers = layers

#         self.has_history = False
#         self.has_val_history = False

#         self.fc_layers = []
#         self.dropouts = []
#         self.activations = []

#         # Add first layer
#         self.nlayers = len(layers)
#         # Add first layer
#         self.fc_layers.append(nn.Linear(nfeats, layers[0]))
#         self.dropouts.append(nn.Dropout(p=p_dropout))
#         self.activations.append(nn.ReLU())
#         # Add further hidden layers
#         for ii in range(1, self.nlayers):
#             self.fc_layers.append(nn.Linear(layers[ii-1], layers[ii]))
#             self.dropouts.append(nn.Dropout(p=p_dropout))
#             self.activations.append(nn.ReLU())
#         # Add output layer
#         self.output_layer = nn.Linear(layers[self.nlayers-1], 3)
#         self.softmax = nn.Softmax(1)        
#         # self.loss_function = torch.nn.CrossEntropyLoss()
#         print('hle')
#         self.loss_function = cl.UncertainLoss(penalty=0.5)

#     def encode(self, x):
#         # First layer 
#         h = self.dropouts[0](self.activations[0](self.fc_layers[0](x)))
#         # Keep going
#         for ii in range(1, self.nlayers):
#             h = self.dropouts[ii](self.activations[ii](self.fc_layers[ii](h)))
#         return h

#     def forward(self, x):
#         h = self.encode(x)
#         output = self.output_layer(h)
#         return output

#     def predict_proba(self, x):
#         x = torch.from_numpy(x)
#         x = Variable(x.float())
#         return self.softmax(self.forward(x)).data.numpy()

#     def compute_loss(self, X, y):
#         y_pred = self.forward(X)
#         loss = self.loss_function(y_pred, y)
#         return loss


# class FlipperDnn(nn.Module):
#     def __init__(self, nfeats, layers, p_dropout=0.0):
#         super(FlipperDnn, self).__init__()
        
#         self.p_dropout = p_dropout
#         self.layers = layers

#         self.has_history = False
#         self.has_val_history = False

#         self.fc_layers = []
#         self.dropouts = []
#         self.activations = []

#         # Add first layer
#         self.nlayers = len(layers)
#         # Add first layer
#         self.fc_layers.append(nn.Linear(nfeats, layers[0]))
#         self.dropouts.append(nn.Dropout(p=p_dropout))
#         self.activations.append(nn.ReLU())
#         # Add further hidden layers
#         for ii in range(1, self.nlayers):
#             self.fc_layers.append(nn.Linear(layers[ii-1], layers[ii]))
#             self.dropouts.append(nn.Dropout(p=p_dropout))
#             self.activations.append(nn.ReLU())
#         # Add output layer
#         self.output_layer = nn.Linear(layers[self.nlayers-1], 4)
#         self.softmax = nn.Softmax(1)        
#         # self.loss_function = torch.nn.CrossEntropyLoss()
        
#         self.loss_function = cl.FlipLoss()

#     def encode(self, x):
#         # First layer 
#         h = self.dropouts[0](self.activations[0](self.fc_layers[0](x)))
#         # Keep going
#         for ii in range(1, self.nlayers):
#             h = self.dropouts[ii](self.activations[ii](self.fc_layers[ii](h)))
#         return h

#     def forward(self, x):
#         h = self.encode(x)
#         output = self.output_layer(h)
#         return output

#     def predict_proba(self, x):
#         N = np.shape(x)[0]
#         x = torch.from_numpy(x)
#         x = Variable(x.float())
#         result = self.forward(x)
#         li = F.softmax(result[:, :3], dim=1)
#         pi = torch.sigmoid(result[:, 3])
#         preds = Variable(torch.zeros((N, 2)))
#         preds[:, 0] += li[:, 0] + (1 - pi) * li[:, 2]
#         preds[:, 1] += li[:, 1] + pi * li[:, 2]
        
#         print(preds)
#         print(li)
#         print(pi)
#         return preds.data.numpy()

#     def compute_loss(self, X, y):
#         y_pred = self.forward(X)
#         loss = self.loss_function(y_pred, y)
#         return loss


class C1dnn(nn.Module):
    """A series of 1d convolutions
    """

    def __init__(self, nfeats):
        super(C1dnn, self).__init__()

        self.has_history = False
        self.has_val_history = False

        self.conv1 = nn.Conv1d(1, 50, 3, stride=2)
        self.conv2 = nn.Conv1d(50, 100, 3, stride=2)
        self.conv3 = nn.Conv1d(100, 200, 3, stride=2)
        self.conv4 = nn.Conv1d(200, 400, 3, stride=2)

        l_out = nfeats
        for ii in range(4):
            l_out = np.floor((l_out / 2) - 0.5)
            l_out = int(l_out)

        self.fc1 = nn.Linear(l_out * 400, 2)

        self.softmax = nn.Softmax(1)

        self.loss_function = torch.nn.CrossEntropyLoss()

    def encode(self, x):
        n, m = x.size()
        x = x.view(n, 1, m)
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        return h4.view(n, -1)

    def forward(self, x):
        h4 = self.encode(x)
        h5 = self.fc1(h4)
        return h5

    def predict_proba(self, x):
        x = torch.from_numpy(x)
        x = Variable(x.float())
        return self.softmax(self.forward(x)).data.numpy()

    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        loss = self.loss_function(y_pred, y)
        return loss


class FilterC1dnn(nn.Module):
    """A series of 1d convolutions
    """

    def __init__(self, nfeats, in_chns, stride=5, kernel_size=11,
                 out_channels=50, padding=0, dilation=1, hidden_size=400):
        super(FilterC1dnn, self).__init__()

        self.in_chns = in_chns
        self.has_history = False
        self.has_val_history = False

        self.conv1 = nn.Conv1d(self.in_chns, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride)
        l_out = int(np.floor((nfeats + 2 * padding -
                              dilation * (kernel_size - 1) - 1) / stride + 1))

        self.fc2 = nn.Linear(l_out * out_channels, hidden_size)
        self.fc2p5 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc3 = nn.Linear(hidden_size, 2)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu2p5 = nn.ReLU()

        self.softmax = nn.Softmax(1)

        self.loss_function = torch.nn.CrossEntropyLoss()

    def encode(self, x):
        # print(type(x))
        n, chns, m = x.size()
        h1 = self.relu1(self.conv1(x))
        h2 = self.relu2(self.fc2(h1.view(n, -1)))
        h2p5 = self.relu2p5(self.fc2p5(h2))
        return h2

    def forward(self, x):
        h2 = self.encode(x)
        h3 = self.fc3(h2)
        return h2

    def predict_proba(self, x):
        x = torch.from_numpy(x)
        x = Variable(x.float())
        return self.softmax(self.forward(x)).data.numpy()

    def compute_loss(self, X, y):
        y_pred = self.forward(X)
        loss = self.loss_function(y_pred, y)
        return loss


class AutoEncoder(nn.Module):
    def __init__(self, nfeats, hidden_size=100, rep_size=30, p=0.0):
        super(AutoEncoder, self).__init__()
        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Linear(nfeats, hidden_size, bias=False)
        self.fc1_bn = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, rep_size)
        self.fc3 = nn.Linear(rep_size, hidden_size, bias=False)
        self.fc3_bn = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, nfeats)
        self.relu = nn.ReLU()
        self.nfeats = nfeats
        self.loss_function = torch.nn.MSELoss()

    def encode(self, x):
        h1 = self.dropout(self.relu(self.fc1_bn(self.fc1(x))))
        h2 = self.fc2(h1)
        return h2

    def decode(self, z):
        h3 = self.dropout(self.relu(self.fc3_bn(self.fc3(z))))
        return self.fc4(h3)

    def forward(self, x):
        z = self.encode(x.view(-1, self.nfeats))
        recon = self.decode(z)
        return recon

    def compute_loss(self, X, y):
        recon = self.forward(X)
        loss = self.loss_function(recon, X)
        return loss


class LstmAutoencoder(nn.Module):
    """An autoencoder using an LSTM
    """

    def __init__(self, nfeats, hidden_size=100):
        super(FilterC1dnn, self).__init__()

        # Get input and hidden sizes
        self.T = nfeats
        self.hidden_size = hidden_size

        self.has_history = False
        self.has_val_history = False

        self.rnn1 = nn.RNN(self.T, hidden_size)
        self.rnn2 = nn.RNN(hidden_size, self.T)
        self.loss_function = torch.nn.MSELoss()

    def encode(self, x):
        z, h = self.rnn1(x)
        return output[-1, :, :]

    def decode(self, z):
        # Repeat z into the right shape
        e = z.repeat(self.nfeats, 1, 1)
        x_hat, h = self.rnn2(x)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def predict_proba(self, x):
        x = torch.from_numpy(x)
        x = Variable(x.float())
        return self.softmax(self.forward(x)).data.numpy()

    def compute_loss(self, x, y):
        x_hat = self.forward(x)
        loss = self.loss_function(x_hat, x)
        return loss