import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

from sklearn.metrics import classification_report

from utils import dev, get_loader, init_weights, train_model, get_acc
from utils import get_predictions
from load_data import append, get_XY

#hyper-parameters
WINDOW = 5
LR = .005

#model architecture
class CNN(nn.Module):
    def __init__(self, conv_dim):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(conv_dim, 64, 9, padding=5)
        self.conv2 = nn.Conv1d(64, 32, 7, padding=3)
        self.conv3 = nn.Conv1d(32, 64, 7, padding=3)
        self.conv4 = nn.Conv1d(64, 32, 5, padding=2)
        self.conv5 = nn.Conv1d(32, 64, 5, padding=2)
        self.conv6 = nn.Conv1d(64, 32, 3, padding=1)
        self.conv7 = nn.Conv1d(32, 64, 3, padding=1)
        self.conv8 = nn.Conv1d(64, 5, 3)

        self.flat = nn.Flatten()
        self.dense = nn.Linear((WINDOW)*5, 5)

        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)

        residual = out
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        out = self.relu(out)
        residual = out

        out = self.conv4(out)
        out = self.relu(out)
        out = self.conv5(out)
        out += residual
        out = self.relu(out)
        residual = out

        out = self.conv6(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.relu(out)
        out = self.conv8(out)

        out = self.flat(out)
        out, self.dense(out)

        return out

def train_and_test(i, X_l, y_l):
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    for j in range(len(X_l)):
        if i==j:
            X_test = X_l[j]
            y_test = y_l[j]
        else:
            X_train = append(X_train, X_l[j])
            y_train = append(y_train, y_l[j])
    return X_train, X_test, y_train, y_test


def cross_val(X, y):
    chunk_size = X.shape[0]//5
    X_chunks = []
    y_chunks = []
    for i in range(4):
        X_chunks.append(X[i*chunk_size:(i+1)*chunk_size,...])
        y_chunks.append(y[i*chunk_size:(i+1)*chunk_size,...])
    X_chunks.append(X[(i+1)*chunk_size:,...])
    y_chunks.append(y[(i+1)*chunk_size:,...])

    accs = []
    for i in range(5):
        X_train, X_test, y_train, y_test = train_and_test(i, X_chunks, y_chunks)
        train_loader = get_loader(dev, X_train[:,None,:], y_train)
        test_loader = get_loader(dev, X_test[:,None,:], y_test)
        model = CNN(1)
        train_model(model, train_loader, 300, True)
        preds = get_predictions(test_loader, model, dev)
        acc = get_acc(test_loader, model, dev)
        print(acc)
        accs.append(acc)
        print(classification_report(preds, y_test))
    print(accs)
    return accs

X, y = get_XY()
accs = cross_val(X, y)
print(np.mean(accs))
