import pandas as pd

train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")

train_df.to_pickle('./data/train.pkl')
test_df.to_pickle('./data/test.pkl')