import os
from collections import defaultdict
import sys
from MNIST_data_intake import MNISTDataInjest
from logistic_regression_tob import LogisticRegressionModel
import numpy as np
from functools import partial
import argparse
import json
from Utils.CalculateStats import CalculateStats
from MNIST_models import MNISTModels


sys.set_int_max_str_digits(0)


mode = "hidden_layer"
# "single_layer", "hidden_layer", "pytorch"

def main():
    
    ls, ws, bs, probs = os.listdir(path_dir), defaultdict(int), defaultdict(int), defaultdict(int)
    for dir in ls:
        if "labels" in dir:
            MNIST.injest_labels(path_dir, byte_num, dir)
        elif "images" in dir:
            MNIST.injest_images(path_dir, dir)
    for key in imgs:
        # flatten images
        imgs[key] = np.reshape(imgs[key], (imgs[key].shape[0], matr_size))
        
    # train logistic regressions
    if mode=="single_layer":
        res, stats = MNIST_models.train_and_test_single_layer(imgs, labs, path_dir)
        
    elif mode=="hidden_layer":
                
        network = MNIST_models.train_load_hidden_layer(imgs, labs, opt.model_path, opt.model_dir)
        res, stats = MNIST_models.test_hidden_layer(imgs, labs, network, test_size=10000) 
    accuracy, total_accuracy = calculate_stats.calculate_accuracy(stats)
    calculate_stats.print_stats(accuracy, total_accuracy, stats)
     
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/thomasobrien/dev/src/Machine-Learning-Foundations/data/MNIST_data", help='what sort of augmentation')
    parser.add_argument('--model_path', type=str, default="/home/thomasobrien/dev/src/Machine-Learning-Foundations/models", help='what sort of augmentation')
    opt = parser.parse_args()
    
    imgs, labs = defaultdict(partial(np.ndarray, 0)), defaultdict(partial(np.ndarray, 0))
    path_dir = opt.path
    byte_num, matr_size, cat_num = 1, 784, 10
    cats = [float(x) for x in range(10)]
    MNIST = MNISTDataInjest(imgs, labs)
    calculate_stats = CalculateStats(cats)
    MNIST_models = MNISTModels(MNIST, cats, cat_num, matr_size)
    opt.model_dir = MNIST_models.get_unique_directory(opt.model_path, "run")
    main()
    
    