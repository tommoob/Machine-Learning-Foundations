import os
from collections import defaultdict
import sys
from MNIST_data_intake import MNISTDataInjest
from logistic_regression_tob import LogisticRegressionModel
import numpy as np
from functools import partial
import json


sys.set_int_max_str_digits(0)

test_image_string, test_label_string = "t10k-images-idx3-ubyte", 't10k-labels-idx1-ubyte'
train_image_string, train_label_string = 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'

def main():
    load_parameters = True
    
    ls, ws, bs, probs = os.listdir(path_dir), defaultdict(int), defaultdict(int), defaultdict(int)
    for dir in ls:
        print(dir)
        if "labels" in dir:
            MNIST.injest_labels(path_dir, byte_num, dir)
        elif "images" in dir:
            MNIST.injest_images(path_dir, dir)
    for key in imgs:
        # flatten images
        imgs[key] = np.reshape(imgs[key], (imgs[key].shape[0], matr_size))
    # train logistic regressions
    if not load_parameters:
        ws, bs = train_logistic_models(ws, bs, learn_rate=0.001, nepochs=100)
        
    else:
        ws = np.load(os.path.join(path_dir, "ws.json.npz"))
        bs = np.load(os.path.join(path_dir, "bs.json.npz"))

    for jx in range(10):
        probs[str(jx)] = logistic_regression_model.predict_proba(imgs[test_image_string].T, ws[str(jx)], bs[str(jx)])
        
    for ex in range(labs[test_label_string].size):
        target, high_cat, high_prob = find_highest_prob(labs, probs, ex)
        if not high_cat==[]:
            print(f"For image {ex}, the target was {target}, while the highest probability was for {high_cat} with a probability of {high_prob}")
      
      
def train_logistic_models(ws, bs, learn_rate=0.01, nepochs=100):
    for ix in range(10):
        print(f"training the {ix} model")
        spec_labs = MNIST.select_for_integer(labs[train_label_string], ix)
        ws[str(ix)], bs[str(ix)] = logistic_regression_model.logistic_fit(imgs[train_image_string].T, spec_labs, nepochs=nepochs, learn_rate=learn_rate)

    np.savez(os.path.join(path_dir, "ws.json"), **ws)
    np.savez(os.path.join(path_dir, "bs.json"), **bs)
    return ws, bs
      
      
def find_highest_prob(labs, probs, ex):
    target, high_prob, high_cat = labs[test_label_string][ex], [0.001], []
    for key in probs:
        if probs[key][0][ex] > max(high_prob) and not key=="0":
            high_prob.append(probs[key][0][ex])
            high_cat.append(key) 
        """
        if probs[key][0][ex] > 0.01:
            high_prob.append(probs[key][0][ex])
            high_cat.append(key) 
        """
    return target, high_cat, high_prob
            
if __name__ == "__main__":
    imgs, labs = defaultdict(partial(np.ndarray, 0)), defaultdict(partial(np.ndarray, 0))
    path_dir = "/home/thomasobrien/dev/src/Machine-Learning-Foundations/data/MNIST_data" 
    byte_num, matr_size = 1, 784
    MNIST = MNISTDataInjest(imgs, labs)
    logistic_regression_model = LogisticRegressionModel(np.random.randn(matr_size) * np.sqrt(2/matr_size))
    main()
    
    