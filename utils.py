import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

dev = torch.device('cpu')
BATCH_SIZE = 1024
BATCH_SIZE_TRAIN = 1024

def get_loader(dev, X, y):
    print(X.shape)
    target = torch.tensor(y)
    target = target.type(torch.LongTensor)
    data = torch.tensor(X).float()
    tensor = torch.utils.data.TensorDataset(data.to(dev), target.to(dev))
    loader = torch.utils.data.DataLoader(dataset=tensor, batch_size=BATCH_SIZE, shuffle=False,
                                        pin_memory=True)
    return loader

def init_weights(m):
    if type(m)==nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def train_model(curr_model, Loader, epochs=100, verbose=True, rnn=False, lr=.005):

    optimizer = torch.optim.AdamW(curr_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    curr_model.apply(init_weights)

    loss_hist=[]
    acc_hist=[]

    for e in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for params, labels in Loader:
            labels = labels.squeeze().type(torch.LongTensor)
            optimizer.zero_grad()
            prediction = curr_model(params.to(dev))
            class_preds = torch.argmax(prediction, dim=1)
            correct += torch.sum(class_preds == labels)
            loss = loss_func(prediction, labels)
            total += labels.shape[0]
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss = running_loss/len(Loader)
        acc = correct / total
        loss_hist.append(loss)
        acc_hist.append(acc)
        if verbose and (e%20==0 or e==epochs-1):
            print(f'EPOCH:{e}')
            print(f'  Training loss: {loss}')
            print(f'  Accuracy: {acc}')
    return [acc_hist, loss_hist]

def get_acc(Loader, model, dev):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for params, labels in Loader:
            preds = model(params.to(dev))
            class_preds = torch.argmax(preds, dim=1)
            correct += torch.sum(class_preds == labels)
            total = len(labels)
    return correct/total

def get_predictions(Loader, model, dev):
    model.eval()
    y = None
    with torch.no_grad():
        for params, labels in Loader:
            preds = model(params.to(dev))
            class_preds = torch.argmax(preds, dim=1)
            preds = class_preds.detach().cpu().numpy()
            y = append(y, preds)
    return y
