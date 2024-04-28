import csv
import string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import random


class SentimentDataIntake():
    def __init__(self, data_path):
        self.data_path = data_path
        
    def injest_tweets(self):
        rows = [] 

        with open(self.data_path, 'r', encoding='utf-8', errors='replace') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                # Process each row
                rows.append(row)

        return rows
    
    def clean_data(self, tweets):
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words, res_list, lab_list = set(stopwords.words('english')), [], []
        
        for tweet_object in tweets:
            tweet, label = tweet_object[5], tweet_object[0]
            clean_tweet = tweet.translate(str.maketrans('', '', string.punctuation))
            
            word_tokens = word_tokenize(clean_tweet)[1:]
            filtered_sentence = [w.lower() for w in word_tokens if not w.lower() in stop_words]
            res_list.append([label, filtered_sentence])
        
        return res_list
    

    def data_label_split(self, data):

        data, labs = [], [] 
        for entry in data:
            data.append(entry[1])
            labs.append(entry[0])
            
        return data, labs


    def train_test_split(self, clean_tweets):
        train_set, test_set = defaultdict(list), defaultdict(list)
        
        tweet_dict = defaultdict(list)
        for tweet_object in clean_tweets:
            tweet_dict[tweet_object[0]].append(tweet_object[1])
            
        for key in tweet_dict:
            testset_size = len(tweet_dict[key]) // 10            
            random.shuffle(tweet_dict[key])
            train_set[key] = (tweet_dict[key][testset_size:])
            test_set[key] = (tweet_dict[key][:testset_size])

        train_set, test_set = self.flatten_tweet_dict(train_set), self.flatten_tweet_dict(test_set)
        
        return train_set, test_set
    
    def flatten_tweet_dict(self, dict):
        res_array = []
        for key in dict:
            for tweet in dict[key]:
                res_array.append([key, tweet])
        
        return res_array
            
    