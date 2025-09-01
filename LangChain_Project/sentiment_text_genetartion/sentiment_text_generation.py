import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()   

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
from transformers import pipeline
import torch

classifier = pipeline('sentiment-analysis')
type (classifier)

classifier('I love you')
classifier(['I love you', 'I hate you'])

customer_reviews = pd.read_csv('customers-100.csv')
customer_reviews.head(10)

airline_tweets = pd.read_csv('tweets.csv')
airline_tweets.head(10)

df = airline_tweets[['text', 'airline_sentiment']]
df.head(10)

sns.countplot(df, x='airline_sentiment', palette='viridis')
plt.xlabel('airline_sentiment')
plt.ylabel('count')
plt.show()

df = df[df['airline_sentiment'] != 'neutral']
df = df['target'] = df['airline_sentiment'].map({'negative': 0, 'positive': 1})
print("number of rows:", df.shape[0])


texts = df['text'].tolist()
predictions = classifier(texts)
predictions [:5]

prods = [pred['score'] if pred ['label'].startswith('P') else 1 - pred['score'] for pred in predictions]
preds = np.array ([1 if pred['label'].startswith('P') else 0 for pred in predictions])






  