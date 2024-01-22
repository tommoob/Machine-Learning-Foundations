from sentiment_data import SentimentDataIntake

data_path = "/home/thomasobrien/dev/src/own/data/tweet_emotions.csv"

def main():
    tweets = data_intake.injest_tweets()
    cleaned_tweets = data_intake.clean_data(tweets)
    train_tweets, test_tweets = data_intake.train_test_split(cleaned_tweets)
    print("aaaa")

if __name__ == "__main__":
    data_intake = SentimentDataIntake(data_path=data_path)
    main()