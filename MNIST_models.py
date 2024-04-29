import numpy as np
import os
from logistic_regression_tob import LogisticRegressionModel
from MNIST_hidden_layer import MNISTHiddenLayer
from datetime import datetime
from Utils.CalculateStats import CalculateStats


test_image_string, test_label_string = "t10k-images-idx3-ubyte", 't10k-labels-idx1-ubyte'
train_image_string, train_label_string = 'train-images-idx3-ubyte', 'train-labels-idx1-ubyte'


class MNISTModels():
    def __init__(self, MNIST, cats, cat_num, matr_size):
        self.logistic_regression_model = LogisticRegressionModel(w=np.random.randn(cat_num, matr_size) * np.sqrt(2/(matr_size)), 
                                                                 b=np.random.randn(cat_num) * np.sqrt(2/cat_num))
        self.MNIST = MNIST
        self.calculate_stats = CalculateStats(cats)
        self.MNIST_hidden_layer = MNISTHiddenLayer()
        
        
    def train_single_layer_logistic_models(self, labs, imgs, train_image_string, train_label_string, path_dir, learn_rate=0.001, nepochs=100):
        spec_labs = self.MNIST.one_hot_labels(labs[train_label_string])
        w, b = self.logistic_regression_model.logistic_fit(imgs[train_image_string].T, spec_labs, nepochs=nepochs, learn_rate=learn_rate)

        np.savetxt(os.path.join(path_dir, "ws.json"), w, fmt='%d')
        np.savetxt(os.path.join(path_dir, "bs.json"), b, fmt='%d')
        return w, b
    
    
    def train_and_test_single_layer(self, imgs, labs, path_dir, load_parameters=False):

        if not load_parameters:
            w, b = self.train_single_layer_logistic_models(labs, imgs, train_image_string, train_label_string, path_dir, learn_rate=0.05, nepochs=50)
            
        else:
            w = np.loadtxt(os.path.join(path_dir, "ws.json.npz"), dtype=int)
            b = np.loadtxt(os.path.join(path_dir, "bs.json.npz"), dtype=int)

        probs = self.logistic_regression_model.predict_proba(imgs[test_image_string].T, w, b).T
            
        res, stats = self.calculate_stats.find_highest_prob(probs, labs, test_label_string)
        
        return res, stats


    def train_load_hidden_layer(self, data, labs, model_path, new_model_directory, n_outputs=10, train_size=10000, n_epochs=50, learning_rate=0.01, clip_threshold=10000, n_hidden=20, batch_size=200):
        model_load_path = os.path.join(model_path, f"MNIST_model_hidden_layer_{n_hidden}_epochs_{n_epochs}_train_size_{train_size}_learning_rate_{learning_rate}.pkl")
        model_save_path = os.path.join(new_model_directory, f"MNIST_model_hidden_layer_{n_hidden}_epochs_{n_epochs}_train_size_{train_size}_learning_rate_{learning_rate}.pkl")

        
        if not os.path.exists(model_load_path):
            os.makedirs(model_save_path, exist_ok=True)
            network = self.MNIST_hidden_layer.initialize_network(n_hidden=n_hidden)
            self.MNIST_hidden_layer.train_network(network, data[train_image_string][:train_size], labs[train_label_string][:train_size], learning_rate, n_epochs, n_outputs, batch_size, clip_threshold, model_save_path)
            self.MNIST_hidden_layer.save_network_weights(network, model_load_path)
        else:
            print("loading model")
            model_save_path = os.path.join(model_path, f"MNIST_model_hidden_layer_{n_hidden}_epochs_{n_epochs}_train_size_{train_size}_learning_rate_{learning_rate}.pkl")
            network = self.MNIST_hidden_layer.load_network_weights(model_save_path)
        return network
        
    
    def test_hidden_layer(self, data, labs, network, test_size=10000):
        preds = np.zeros((test_size, 10))
        for i, lab in enumerate(labs[test_label_string][:test_size]):
            preds[i] = self.MNIST_hidden_layer.predict(network, data[test_image_string][i])
        
        res, stats = self.calculate_stats.find_highest_prob(preds, labs, test_label_string)
        
        return res, stats
    
    
    def get_unique_directory(self, path, directory):
        counter = 1
        unique_directory = os.path.join(path, directory)

        while os.path.exists(unique_directory):
            unique_directory = os.path.join(path, directory + str(counter))
            counter += 1
        
        return unique_directory 