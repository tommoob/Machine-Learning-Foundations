from collections import defaultdict
import numpy as np


class CalculateStats():
    
    
    def __init__(self, cats):
        self.cats = cats
        
        
    def calculate_precision_recall(self, stats):
        total_correct, precision_sum, recall_sum = 0, 0, 0
        precisions, recalls = defaultdict(int), defaultdict(int)
        for key in self.cats:
            precisions[key] = stats["correct_predictions"][key] / (stats["correct_predictions"][key] + stats["false_positives"][key])
            recalls[key] = stats["correct_predictions"][key] / (stats["correct_predictions"][key] + stats["false_predictions"][key])    
            precision_sum += precisions[key]
            recall_sum += recalls[key]
            
        total_precision = precision_sum / len(self.cats)
        total_recall = recall_sum / len(self.cats)
        
        return precisions, recalls, total_precision, total_recall    
       
      
    def calculate_accuracy(self, stats):
        
        accuracies = defaultdict(int)
        total_correct, total_wrong = 0, 0
        
        for key in stats["correct_predictions"]:
            accuracies[key] = stats["correct_predictions"][key] / (stats["correct_predictions"][key] + stats["false_predictions"][key])
            total_correct += stats["correct_predictions"][key]
            total_wrong += stats["false_predictions"][key]
        total_accuracy = total_correct / (total_correct + total_wrong)
        
        return accuracies, total_accuracy
      
    def find_highest_prob(self, probs, labs, test_label_string):
        res = np.zeros((2, probs.shape[0]))
        # correct_predictions are equivalent to true positives
        # false_predictions are equivalent to false negatives
        correct_predictions, false_predictions, true_negatives, false_positives = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
        stats = {"correct_predictions": correct_predictions, "false_predictions": false_predictions, 
                "true_negatives": true_negatives, "false_positives": false_positives}
        
        for ix in range(probs.shape[0]):
            target = labs[test_label_string][ix]
            res[0, ix] = np.max(probs[ix])
            res[1, ix] = np.where(probs[ix]==np.max(probs[ix]))[0][0]
            
            if int(target) == int(res[1, ix]): 
                stats["correct_predictions"][target] += 1
                true_neg_cats = [x for x in self.cats if x != target]
                for n_key in true_neg_cats:
                    stats["true_negatives"][n_key] += 1
            else:
                stats["false_predictions"][target] += 1
                stats["false_positives"][res[1, ix]] += 1
                
        return res, stats
    
    
    def print_stats(self, accuracy, total_accuracy, stats):
        """print("Total accuracy: ", total_accuracy)
        for key in accuracy:
            print(f"Accuracy of {key} detection is {accuracy[key]}")
        """
        precisions, recalls, total_precision, total_recall = self.calculate_precision_recall(stats)
        print(f"The total precision for the model is {total_precision}")
        print(f"The total recall for the model is {total_recall}")
        """
        for key in precisions:
            print(f"Precision of {key} detection is {precisions[key]}")
            print(f"Recall of {key} detection is {recalls[key]}")   
        """