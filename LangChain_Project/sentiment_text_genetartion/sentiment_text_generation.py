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
df['target'] = df['airline_sentiment'].map({'negative': 0, 'positive': 1})
print("number of rows:", df.shape[0])

texts = df['text'].tolist()
predictions = classifier(texts)
predictions [:5]

probs = [pred['score'] if pred ['label'].startswith('P') else 1 - pred['score'] for pred in predictions]
preds = np.array([1 if pred['label'].startswith('P') else 0 for pred in predictions])
print(f"Accuracy: {round(np.mean(df['target'] == preds) * 100, 2)} %")

cm = confusion_matrix(df['target'], preds, normalize='true')
def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.4)
    sns.heatmap(cm, annot=True, fmt = 'g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



plot_confusion_matrix(cm, ['negative', 'positive'])

print (f"roc auc_score: {roc_auc_score(df['target'], probs)}")

poems = pd.read_csv('robert_frost.csv')
poems.head(10)
content = poems['Content'].dropna().tolist()
lines = [ ]
for poem in content:
    for line in poem.split('\n'):
        lines.append(line.strip())
            
    lines = [line for line in lines if len(line) > 0]
    lines[:5]

gen = pipeline('text-generation')
lines[0]

gen(lines[0], max_length=20)





  