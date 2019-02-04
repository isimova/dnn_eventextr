#!/usr/bin/env python
# -*- coding: utf-8 -*-
from evaluate_nn import evaluate
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import nn
import torch.nn.functional as F
import torch
import sys

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHT_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
GLOVE = "glove.6B.300d.txt"


def embed_glove(sentence, glove):
    """
    Returns a glove sentence embedding by averaging the glove embeddings of
    all tokens in the sentence

    :param sentence: list of tokens
    :param glove: glove embedding dictionary
    :return: sentence embedding
    """
    glove_sent_emb=[]
    for t in sentence:
        if t.lower() in glove:
            glove_sent_emb.append(glove[t.lower()])
        else:
            glove_sent_emb.append(glove["unk"])
    return torch.tensor(glove_sent_emb).cuda().mean(0)


def load_glove():
    """
    Loads glove embedding

    :return: glove embedding dict
    """
    print ("Loading glove word embeddings file..")
    word_to_wordemb = {}
    with open(GLOVE, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip().split()
            word_to_wordemb[line[0]] = [float(v) for v in line[1:]]
    return word_to_wordemb


def read_file(fname):
    """
    Reads a file name containing one token per line with sentences
    seperated by empty line

    :param fname: file name to read
    :return: sentence embeddings and event class ids
    """
    sentences = []
    events = []
    sentence = []
    event = []
    with open(fname, "r", encoding="utf-8") as fin:
        for line in fin:
            if len(line.strip()) == 0 and sentence:
                sentences.append(sentence)
                events.append(torch.tensor(event).cuda().long().view(1, -1)[0])
                sentence = []
                event = []
            else:
                line = line.strip().split("\t")
                sentence.append(line[0])
                event.append(int(line[-1]))
    return sentences, events


class MyModel(nn.Module):

    def __init__(self, emb_dim, hidden_size, out_dim):
        """
        Initialize a custom net which embeds input sentences with elmo to allow
        for the tuning of elmo parameters

        :param emb_dim: elmo embedding dim + sentence emb dim
        :param hidden_size: number of neurons in hidden layer
        :param out_dim: output dimension, i.e. number of classes
        """
        super(MyModel, self).__init__()
        self.elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHT_FILE, 1)
        self.linear1 = nn.Linear(emb_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, out_dim)

    def forward(self, sentence, sent_emb):
        character_ids = batch_to_ids(sentence)
        elmo_emb = self.elmo(character_ids.cuda())["elmo_representations"][0]
        embeddings = torch.cat((elmo_emb, sent_emb.repeat(elmo_emb.shape[0], 1)), 1)
        h_relu = F.relu(self.linear1(embeddings))
        y_pred = self.linear2(h_relu)
        return y_pred[0]


def run_exp(train_file, test_file, dev_file):

    # read in data
    test_X, test_Y = read_file(test_file)
    train_X, train_Y = read_file(train_file)
    dev_X, dev_Y = read_file(dev_file)

    params = {
        "embedding_dim": 1324,  # standard elmo embedding dimension
        "num_classes": 34,  # output dimension: 33 event classes + None class
        "num_epochs": 20,
        "learning_rate": 0.001
    }
    params["hidden_size"] = (params["embedding_dim"] +
                             params["num_classes"])//2
    print("Parameters: " + str(params))

    # initialize model
    model = MyModel(params["embedding_dim"],
                    params["hidden_size"],
                    params["num_classes"])
    model = model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

    glove = load_glove()

    # sanity check to see if parameters are modified
    with open("elmo_params_before_training.out", "w") as fo:
        for name, param in model.elmo.named_parameters():
            fo.write("Param name: " + name + "\n")
            fo.write("Param value: " + str(param.data) + "\n\n")

    for t in range(params["num_epochs"]):
        avg_loss = 0
        avg_loss_dev = 0

        # train
        for sent_id in range(len(train_X)):
            model.train()
            sent = [train_X[sent_id]]
            sent_emb = embed_glove(train_X[sent_id], glove)
            events = train_Y[sent_id]
            preds = model(sent, sent_emb)
            loss = loss_fn(preds, events)
            avg_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # run on dev to monitor overfitting
        model.eval()
        for sent_id in range(len(dev_X)):
            sent = [dev_X[sent_id]]
            sent_emb = embed_glove(dev_X[sent_id], glove)
            preds = model(sent, sent_emb)
            events = dev_Y[sent_id]
            loss_dev = loss_fn(preds, events)
            avg_loss_dev += loss_dev.item()

        print("Epoch: {}\tAvg TRAIN Loss: {:.4f}\tDEV Loss: {:.4f}".format(t, avg_loss/len(train_X), avg_loss_dev/len(dev_X)))

    # sanity check to see if parameters are modified
    with open("elmo_params_after_training.out", "w") as fo:
        for name, param in model.elmo.named_parameters():
            fo.write("Param name: " + name + "\n")
            fo.write("Param value: " + str(param.data) + "\n\n")

    print("TEST")
    model.eval()
    output = []
    gold_Y = []
    for sent_id in range(len(test_X)):
        sent = [test_X[sent_id]]
        sent_emb = embed_glove(test_X[sent_id], glove)
        preds = model(sent, sent_emb)
        for p in preds:
            output.append(p)
        for e in test_Y[sent_id]:
            gold_Y.append(e)

    output = torch.stack(output)
    gold_Y = torch.stack(gold_Y)
    evaluate(gold_Y, output)


if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python3 relu_ffnn_elmo_trainable.py <train file> <test file> <dev file>")
        sys.exit(0)

    run_exp(sys.argv[1], sys.argv[2], sys.argv[3])
