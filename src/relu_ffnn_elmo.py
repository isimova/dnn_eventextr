#!/usr/bin/env python
# -*- coding: utf-8 -*-
from evaluate_nn import evaluate
import torch
import sys

SAVE_MODEL=True

def read_file(fname):
    dat = []
    with open(fname, "r") as fin:
        for line in fin:
            line = line.strip().split()
            dat.append([float(v) for v in line])
    return dat

def run_exp(train_file, test_file, dev_file):

    test_data = torch.tensor(read_file(test_file)).cuda()
    train_data = torch.tensor(read_file(train_file)).cuda()
    dev_data = torch.tensor(read_file(dev_file)).cuda()

    '''data format:
    token emb: data[0:1024]
    sentence emb: data[1024:1324]
    preds colum: data[1324]
    events column: data[1325]'''

    # train
    X = train_data[:, :1325]
    Y = train_data[:, 1325].long().view(1, -1)[0]

    # test
    X_test = test_data[:, :1325]
    Y_test = test_data[:, 1325].long().view(1, -1)[0]

    # dev
    X_dev = dev_data[:, :1325]
    Y_dev = dev_data[:, 1325].long().view(1, -1)[0]

    params = {
        "embedding_dim": X.shape[1],  # input embedding dimension
        "num_classes": 34,  # output dimension: 33 event classes + None class
        "num_epochs": 300,
        "learning_rate": 0.001
    }
    params["hidden_size"] = (params["embedding_dim"] +
                             params["num_classes"])//2
    print("Parameters: " + str(params))

    model = torch.nn.Sequential(
        torch.nn.Linear(params["embedding_dim"], params["hidden_size"]),
        torch.nn.ReLU(),
        torch.nn.Linear(params["hidden_size"], params["num_classes"]),
    )
    model = model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    for t in range(params["num_epochs"]):
        model.train()
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        Y_dev_pred = model(X_dev)
        loss_dev = loss_fn(Y_dev_pred, Y_dev)
        print("Epoch: {}\tAvg TRAIN Loss: {:.4f}\tAvg DEV loss: {:.4f}".format(
            t, loss.item(), loss_dev.item()))

    if SAVE_MODEL:
        torch.save(model.state_dict(), "model.elmo.pt")

    model.eval()

    print("TEST")
    evaluate(Y_test, model(X_test), tuning=True)

    print("\nDEV")
    evaluate(Y_dev, model(X_dev))


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python3 relu_ffnn_elmo.py <train file> <test file> <dev file>")
        sys.exit(0)

    run_exp(sys.argv[1], sys.argv[2], sys.argv[3])
