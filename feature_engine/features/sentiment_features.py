# imports
import numpy as np
import pandas as pd
from scipy.special import softmax
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

class sentiment_model:
    def __init__(self, model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.pipeline = pipeline('sentiment-analysis', model=self.model, tokenizer=self.tokenizer, config=self.config)

    def encode_input(self, text):
        return self.tokenizer(text, return_tensors='pt')
    
    def get_score(self, text, cluster):
        encoded_input = self.encode_input(text)      
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        result = {}
        for i in range(scores.shape[0]):
            l = self.config.id2label[ranking[i]]
            s = scores[ranking[i]]
            result[l] = s
            print(f"{i+1}) {l} {np.round(float(s), 4)}")
        new_result = self.adjust_score(result, cluster)
        return result, new_result
    
    def adjust_score(self, result, cluster):
        new_result = result.copy()
        if cluster == 0:
            new_result['positive'] *= 1.2
            new_result['negative'] *= 0.8
        elif cluster == 1:
            new_result['positive'] *= 0.8
            new_result['negative'] *= 1.2
        elif cluster == 2:
            new_result['negative'] *= 0.9
        elif cluster == 3:
            new_result['positive'] *= 0.9
        return new_result
print(0)
# model0 = sentiment_model()

# TODO - DEFINE YOUR FEATURE EXTRACTOR HERE
def get_sentiment(model, text, cluster):
    return model.get_score(text, cluster)

def normalization(chat_data):
    df = chat_data.copy()
    df = df.dropna()
    df = df.reset_index()
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    standard_scaler = preprocessing.StandardScaler()
    features = ['efficiency', 'score', 'duration']
    df_scaled = standard_scaler.fit_transform(df[features])
    df_scaled_0 = pd.DataFrame(df_scaled, columns=features)
    df_combined = pd.concat([df.drop(columns=features), df_scaled_0], axis=1)
    return df_scaled, df_combined

def clustering(df_scaled, df_combined):
    wcss = []
    silhouette_coefficients = []
    features = ['efficiency', 'duration','score']
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(df_scaled, kmeans.labels_)
        silhouette_coefficients.append(score)

        # plt.figure(figsize=(6, 3))
        # plt.plot(range(2, 11), wcss, marker='o')
        # plt.title('Elbow Method')
        # plt.xlabel('Number of clusters')
        # plt.ylabel('WCSS')
        # plt.show()

        # plt.figure(figsize=(6, 3))
        # plt.plot(range(2, 11), silhouette_coefficients, marker='o')
        # plt.title('Silhouette Coefficients')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Silhouette Coefficient')
        # plt.show()

        k = 4
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        df_combined['cluster'] = kmeans.fit_predict(df_scaled)
    
    cluster_characteristics = df_combined.groupby('cluster')[features].mean()
    print(cluster_characteristics)
    return df_combined


