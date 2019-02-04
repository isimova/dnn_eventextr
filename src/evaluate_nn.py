#!/usr/bin/env python
# coding: utf-8
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
import numpy as np
import torch


def get_preds(pred_softmax, theta=0):
    """
    Compute class predictions from a softmax distribution. Theta is
    a confidence value: a prediction is returned iff it's probability
    is larger than the next largest class probability by theta.

    :param pred_softmax: softmax tensor of shape (num_examples, num_classes)
    :param theta: confidence parameter
    :return: class predictions
    """
    if theta == 0:
        return torch.max(pred_softmax, 1)[1]  # ids of max values
    else:
        preds=[]
        for p in pred_softmax:
            top = torch.topk(p, 2)  # get the top two max values
            if top[0][0]-theta > top[0][1]:
                preds.append(top[1][0].item())
            else:
                preds.append(-1)  # return None class if confidence is low
        return torch.tensor(preds).cuda()


def get_micro_f1(gold_Y, pred_Y, labels):
    return 100*precision(gold_Y, pred_Y, labels, average="micro"), \
           100*recall(gold_Y, pred_Y, labels, average="micro"), \
           100*f1(gold_Y, pred_Y, labels, average="micro")


def get_f1(gold_Y, pred_Y, all_class=False):
    """
    Print out micro-averaged P/R/F1 scores

    :param gold_Y: gold output
    :param pred_Y: predicted output
    :param all_class: if True also print results for class 0
    :return:
    """
    if all_class:
        return get_micro_f1(gold_Y, pred_Y, labels=np.arange(0, 33))

    return get_micro_f1(gold_Y, pred_Y, labels=np.arange(1, 33))


def get_trigger_identification_f1(gold_Y, pred_Y):
    """
    Print out P/R/F1 scores for trigger identification

    :param gold_Y: gold output
    :param pred_Y: predicted output
    :return:
    """
    gold_ti = []
    pred_ti = []

    for i in range(len(gold_Y)):
        if gold_Y[i] != 0:
            gold_ti.append(1)
        else:
            gold_ti.append(0)
        if pred_Y[i] != 0:
            pred_ti.append(1)
        else:
            pred_ti.append(0)

    return 100*precision(gold_ti, pred_ti), \
           100*recall(gold_ti, pred_ti), \
           100*f1(gold_ti, pred_ti)


def evaluate(gold_Y, pred_Y_softmax, tuning=False):
    """
    Print out trigger identification and trigger classification accuracy

    :param gold_Y: gold output
    :param pred_Y_softmax: softmax prediction of shape (num_examples, num_classes)
    :return: None
    """
    pred_Y = get_preds(pred_Y_softmax)

    p, r, f = get_trigger_identification_f1(gold_Y, pred_Y)
    print("trigger identification P/R/F1: {:.1f}/{:.1f}/{:.1f}".format(p, r, f))

    p, r, f = get_f1(gold_Y, pred_Y)
    print("micro-avg P/R/F1 (positive classes): {:.1f}/{:.1f}/{:.1f}".format(p, r, f))

    correct, all_pos, acc = get_accuracy(gold_Y, pred_Y)
    print("accuracy (positive classes): {}/{} ({:.1f}%)".format(correct, all_pos, acc))

    if tuning:
        theta = 0.0
        print("THETA\tP\tR\tF1")
        while True:
            pred_Y = get_preds(pred_Y_softmax, theta)
            p, r, f = get_f1(gold_Y, pred_Y)
            print("{:.1f}\t{:.1f}\t{:.1f}\t{:.1f}".format(theta, p, r, f))
            if p == 100 or p == 0:
                break
            theta += 0.2
