import pandas as pd

sub = pd.read_csv('dataset/sample_submission.csv')
test = pd.read_csv('dataset/test.csv')

print("Sample submission shape:", sub.shape)
print("Test shape:", test.shape)

print(sub.head())
