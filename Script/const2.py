# Import des bibliothèques nécessaires
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk  # Bibliothèque pour le traitement du langage naturel
import numpy as np
import scipy  # Bibliothèque scientifique pour Python
import sklearn  # Bibliothèque de machine learning
from nltk.corpus import stopwords
import random
random.seed(42)
import math

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Découpage des données en 3 set : train(80%), test(10%), dev(10%)
id_liste = [item["id"] for item in data]

# Train
size_train = int(len(id_liste) * 0.8)  
train = random.sample(id_liste, size_train)
corpus_train = split(data, train, "./Corpus/train.json")

# Test
remain = list(set(id_liste).difference(train))
test = random.sample(remain, math.ceil((len(id_liste) - len(train)) / 2))
corpus_test = split(data, test, "./Corpus/test.json")

# Dev
dev = list(set(remain).difference(test))
corpus_dev = split(data, dev, "./Corpus/dev.json")

# On recupère le contenu textuel de chaque set
train_txt = [item["content"] for item in corpus_train]
test_txt = [item["content"] for item in corpus_test]
dev_txt = [item["content"] for item in corpus_dev]

# liste de stopwords en français
french_stop_words = stopwords.words('french')
count_vectorizer = CountVectorizer(stop_words=french_stop_words)
tfidf_vectorizer = TfidfVectorizer(stop_words=french_stop_words, max_df=0.7)

# Entraînement des vectorizers sur l'ensemble d'entraînement et transformation 
X_train_count = count_vectorizer.fit_transform(train_txt)
X_test_count = count_vectorizer.transform(test_txt)
X_dev_count = count_vectorizer.transform(dev_txt)

X_train_tfidf = tfidf_vectorizer.fit_transform(train_txt)
X_test_tfidf = tfidf_vectorizer.transform(test_txt)
X_dev_tfidf = tfidf_vectorizer.transform(dev_txt)
# Téléchargez la liste de mots d'arrêt en français si ce n'est pas déjà fait
nltk.download('stopwords')


# ensemble d'entrainement
with open('./Corpus/train.json', 'r', encoding='utf-8') as f_train:
    train_data = json.load(f_train)
with open('./Corpus/dev.json', 'r', encoding='utf-8') as f_dev:
    dev_data = json.load(f_dev)
with open('./Corpus/test.json', 'r', encoding='utf-8') as f_test:
    test_data = json.load(f_test)

# Charger les données JSON
with open('./Data/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convertir les données JSON en DataFrame pandas
df = pd.DataFrame(data['items'])

# Diviser les données en ensembles d'entraînement et de test
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extraire les caractéristiques
# liste de stopwords en français
french_stop_words = stopwords.words('french')
count_vectorizer = CountVectorizer(stop_words=french_stop_words)
tfidf_vectorizer = TfidfVectorizer(stop_words=french_stop_words, max_df=0.7)


# Entraînement des vectorizers sur l'ensemble d'de données et transformation 
# X_train_count = count_vectorizer.fit_transform(train_data['content'])
# X_test_count = count_vectorizer.transform(test_data['content'])
# X_dev_count = count_vectorizer.transform(dev_data['content'])

# X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['content'])
# X_test_tfidf = tfidf_vectorizer.transform(test_data['content'])
# X_dev_tfidf = tfidf_vectorizer.transform(dev_data['content'])

# Définir les étiquettes
y_train = train_data['rating']
y_test = test_data['rating']

# Entraîner le modèle Naive Bayes
nb_classifier = MultinomialNB()

# Entraîner le modèle sur les caractéristiques TF-IDF
nb_classifier.fit(X_train_tfidf, y_train)

# Prédiction sur les données de test
y_pred = nb_classifier.predict(X_test_tfidf)

# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
