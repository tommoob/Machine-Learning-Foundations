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
    load_parameters = False
    
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
    if not load_parameters:
        w, b = train_logistic_models(learn_rate=0.075, nepochs=100)
        
    else:
        w = np.loadtxt(os.path.join(path_dir, "ws.json.npz"), dtype=int)
        b = np.loadtxt(os.path.join(path_dir, "bs.json.npz"), dtype=int)

    probs = logistic_regression_model.predict_proba(imgs[test_image_string].T, w, b)
        
    res, stats = find_highest_prob(probs)
    
    accuracy, total_accuracy = calculate_accuracy(stats)
    print("Total accuracy: ", total_accuracy)
    for key in accuracy:
        print(f"Accuracy of {key} detection is {accuracy[key]}")

    precisions, recalls = calculate_precision_recall(stats)
    for key in precisions:
        print(f"Precision of {key} detection is {precisions[key]}")
        print(f"Recall of {key} detection is {recalls[key]}")

      
def calculate_precision_recall(stats):
    precisions, recalls = defaultdict(int), defaultdict(int)
    for key in cats:
        precisions[key] = stats["correct_predictions"][key] / (stats["correct_predictions"][key] + stats["false_positives"][key])
        recalls[key] = stats["correct_predictions"][key] / (stats["correct_predictions"][key] + stats["false_predictions"][key])
    return precisions, recalls    
       
      
def calculate_accuracy(stats):
    
    accuracies = defaultdict(int)
    total_correct, total_wrong = 0, 0
    
    for key in stats["correct_predictions"]:
        accuracies[key] = stats["correct_predictions"][key] / (stats["correct_predictions"][key] + stats["false_predictions"][key])
        total_correct += stats["correct_predictions"][key]
        total_wrong += stats["false_predictions"][key]
    total_accuracy = total_correct / (total_correct + total_wrong)
    
    return accuracies, total_accuracy
      
      
def train_logistic_models(learn_rate=0.001, nepochs=100):
    spec_labs = MNIST.one_hot_labels(labs[train_label_string])
    w, b = logistic_regression_model.logistic_fit(imgs[train_image_string].T, spec_labs, nepochs=nepochs, learn_rate=learn_rate)

    np.savetxt(os.path.join(path_dir, "ws.json"), w, fmt='%d')
    np.savetxt(os.path.join(path_dir, "bs.json"), b, fmt='%d')
    return w, b
      
           
def find_highest_prob(probs):
    res = np.zeros((2, probs.shape[1]))
    # correct_predictions are equivalent to true positives
    # false_predictions are equivalent to false negatives
    correct_predictions, false_predictions, true_negatives, false_positives = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    stats = {"correct_predictions": correct_predictions, "false_predictions": false_predictions, 
             "true_negatives": true_negatives, "false_positives": false_positives}
    
    for ix in range(probs.shape[1]):
        target = labs[test_label_string][ix]
        res[0, ix] = np.max(probs[:, ix])
        res[1, ix] = np.where(probs[:, ix]==np.max(probs[:, ix]))[0][0]
        #print(f"For image {ix}, the target was {target}, but {res[1, ix]} was predicted with a probability of {res[0, ix]}")
        
        if target == res[1, ix]: 
            stats["correct_predictions"][target] += 1
            true_neg_cats = [x for x in cats if x != target]
            for n_key in true_neg_cats:
                stats["true_negatives"][n_key] += 1
        else:
            stats["false_predictions"][target] +=1
            stats["false_positives"][res[1, ix]] += 1
            
    return res, stats
            
if __name__ == "__main__":
    imgs, labs = defaultdict(partial(np.ndarray, 0)), defaultdict(partial(np.ndarray, 0))
    path_dir = "/home/thomasobrien/dev/src/Machine-Learning-Foundations/data/MNIST_data" 
    byte_num, matr_size, cat_num = 1, 784, 10
    cats = [float(x) for x in range(10)]
    MNIST = MNISTDataInjest(imgs, labs)
    logistic_regression_model = LogisticRegressionModel(w=np.random.randn(cat_num, matr_size) * np.sqrt(2/(matr_size)), b=np.random.randn(cat_num) * np.sqrt(2/cat_num))
    main()
    
    