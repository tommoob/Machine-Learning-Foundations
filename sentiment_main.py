from sentiment_data import SentimentDataIntake
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data_path = "/Users/tob/dev/data/training.1600000.processed.noemoticon.csv"

def main():
    tweets = data_intake.injest_tweets()
    cleaned_tweets = data_intake.clean_data(tweets)
    train_tweets, test_tweets = data_intake.train_test_split(cleaned_tweets)
    X_train, y_train, X_test, y_test = data_intake.data_label_split(train_tweets), data_intake.data_label_split(test_tweets)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict the sentiment labels for the testing set
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


    print("aaaa")

if __name__ == "__main__":
    data_intake = SentimentDataIntake(data_path=data_path)
    main()