import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LogisticRegressionModel:
    
    def __init__(self, w=[0.01, 0.01], b=0.0, var_size=10):
        
        self.coefficient = np.array(w).reshape(var_size, -1).T
        # (10, -1)
        if isinstance(b, np.ndarray):
            self.intercept = np.array(b).reshape(1, -1).T
        else:
            self.intercept = b
        
    def logistic_fit(self, xs, ys, nepochs=10, learn_rate=0.001):
        
        z, m = 0, ys.size
        w, b = self.coefficient, self.intercept
        for epoch in range(nepochs):
            z = np.dot(w.T, xs) + b
            a = 1.0 / (1.0 + np.exp(-z))
            
            js = -((ys * np.log(a) + (1 - ys) * np.log(1 - a)))
            j = js.sum() / m
            
            dl_db = (a - ys)
            dj_dw = np.dot(xs, dl_db.T) / m    
            dj_db = np.sum(dl_db) / m 
            
            w -= learn_rate * dj_dw
            b -= learn_rate * dj_db
            
            # print(f"\nepoch: {epoch}, loss: {j}, b: {b}\n")
        self.coefficient, self.intercept = w, b
        return w, b

    def build_dataset(self, type="vertical"):
        """This creates a dataset for our model to fit to. There are options for a vertical line and a generic straight line

        Args:
            type (str, optional): decides the type of dataset you create. Defaults to "vertical".

        """
        num_data_points = 100
        zeros, ones = [0 for x in range(num_data_points)], [1 for x in range(num_data_points)]

        # for vertical line split
        if type.__eq__("vertical"):
            x_s_0, y_s_0 = [random.randint(1, 6) for x in range(num_data_points)], [random.randint(1, 15) for x in range(num_data_points)]
            x_s_1, y_s_1 = [random.randint(8, 12) for x in range(num_data_points)], [random.randint(1, 15) for x in range(num_data_points)]

        #for straight line split
        if type.__eq__("straight_line"):    
            grad = 3
            x_s_0, x_s_1 = [random.randint(1, 15) for x in range(num_data_points)], [random.randint(1, 15) for x in range(num_data_points)]
            y_s_0, y_s_1 = [random.randint(0, grad) * x for x in x_s_0], [random.randint(grad, 2 * grad) * x for x in x_s_1]

        zeros_list = pd.DataFrame({'label': zeros,
                                'x\'s': x_s_0,
                                'y\'s': y_s_0})
        ones_list = pd.DataFrame({'label': ones,
                                'x\'s': x_s_1,
                                'y\'s': y_s_1})
        
        list_combined = pd.concat([zeros_list, ones_list], ignore_index=True, sort=False)
        
        x_0s = list_combined.loc[list_combined["label"] < 0.5].to_numpy()[:, 1:]
        x_1s = list_combined.loc[list_combined["label"] > 0.5].to_numpy()[:, 1:]
        
        list_combined = list_combined.to_numpy()
        labels = list_combined[:, 0].reshape(1, 2 * num_data_points).astype(int)
        xs = list_combined[:, 1:].T
            
        return xs, labels, x_0s, x_1s
    
 
    
    def plot_logistic_regression(self, xs, x_0s, x_1s, w=-100, b=-100):
        if w == -100:
            w = self.coefficient
        if b == -100:
            b = self.intercept
        
        z = 0.5
        x_start, x_end = xs[0, :].min(), xs[0, :].max()
        slope, intercept = - w[0, 0] / w[1, 0], (z - b) / w[1, 0]
        
        x_coords = np.array([x_start, x_end])
        y_coords = slope * x_coords + intercept 
        
        plt.figure()
        
        plt.scatter(x_0s[:, 0], x_0s[:, 1], color='r')
        plt.scatter(x_1s[:, 0], x_1s[:, 1], color='b')
        plt.plot(x_coords, y_coords, color="purple")
        plt.show()
    
    def predict_proba(self, x, w, b):


        # Compute z = wx + b
        Z = np.dot(w.T, x) + b

        # Compute activation (or yhat)
        A = 1.0 / (1.0 + np.exp(-Z))
        return A
        
        
    


