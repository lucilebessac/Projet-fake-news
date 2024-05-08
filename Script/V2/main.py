#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:22:22 2024

@author: guilhem
"""

#__________MODULES
import random
random.seed(42)
import math
import argparse
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from datastructures import save_json, load_json, get_index, split
import nltk  # Bibliothèque pour le traitement du langage naturel
nltk.download('stopwords')

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer


#__________FUNCTIONS


#__________MODULES
def main():
    parser = argparse.ArgumentParser(description="Ce programme a pour but de tester différentes formes de vectorisation et de models pour la détection de fake-news")
    parser.add_argument(
        "file",
        help="Chemin vers le fichier qui contient le corpus au format json."
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        choices=["yes", "no"],
        default="no",
        help="Supprimer les stopwords de l'analyse."
    )
    parser.add_argument(
        "-v",
        "--vectorize",
        choices=["count", "tfidf", "hash"],
        default="count",
        help="Choisir le vectorizer -> 'count' : CountVectorizer | 'tfidf' : TfidfVectorizer | 'hash' : HashingVectorizer"
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["svc", "multi", "dtree", "rforest"],
        default="svc",
        help="Choisir un modèle : 'svc' pour LinearSVC | 'multi' pour MultinomialNB | 'dtree' pour DecisionTree | 'rforest' pour RandomForest"
    )
    
    args = parser.parse_args()
    
    if args.file.split(".")[-1] == "json":
        path_corpura = args.file
    else :
        print("Le fichier contenant les données doit être au format JSON")
        sys.exit()
    
    # Appel la fonction load_json
    dataset = load_json(path_corpura)
    
    # Appel la fonction get_index
    corpus_index = get_index(dataset, "origin")
    # print(corpus_index)
    
    # Découpage des données en 2 set : train(90%), test(10%)
    id_liste = [item["id"] for item in dataset]
    
    # Train
    size_train = math.ceil(len(id_liste) * 0.9)
    train = random.sample(id_liste, size_train)
    corpus_train = split(dataset, train, "../../Corpus/train.json")
    train_index = get_index(corpus_train, "train")

    # Test
    test = list(set(id_liste).difference(train))
    corpus_test = split(dataset, test, "../../Corpus/test.json")
    test_index = get_index(corpus_test, "test")
    
    # On recupère le contenu textuel de chaque set
    train_txt = corpus_train["content"]
    test_txt = corpus_test["content"]
    
    # On recupère le label "rating" de chaque set
    train_labels = corpus_train["rating"]
    test_labels = corpus_test["rating"]
    
    # Liste de stopwords en français
    # french_stop_words = stopwords.words('french')
    # count_vectorizer = CountVectorizer(stop_words=french_stop_words)
    # tfidf_vectorizer = TfidfVectorizer(stop_words=french_stop_words, max_df=0.7)
    
    # On vectorise selon le vectorizer choisi
    if args.vectorize == "count":
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_txt) # Ajustement sur les données d'entraînement
        X_test = vectorizer.transform(test_txt) # Transformation des données de test en utilisant le même vocabulaire
    elif args.vectorize == "tfidf":
        vectorizer = TfidfVectorizer()
        X_train = vectorizer.fit_transform(train_txt)
        X_test = vectorizer.transform(test_txt)
    elif args.vectorize == "hash":
        vectorizer = HashingVectorizer()
        X_train = vectorizer.transform(train_txt)
        X_test = vectorizer.transform(test_txt)

    # Entrainement du modèle sur le Train selon le modèle choisi
    if args.model == "svc":
        clf = LinearSVC().fit(X_train, train_labels)
    elif args.model == "multi":
        clf = MultinomialNB().fit(X_train, train_labels)
    elif args.model == "dtree":
        clf = DecisionTreeClassifier().fit(X_train, train_labels)
    elif args.model == "rforest":
        clf = RandomForestClassifier().fit(X_train, train_labels)

    #Score
    print("Score : ", clf.score(X_test, test_labels))
    
    #Predict
    print("predictions:", clf.predict(X_test))
    
    #Donnee
    print("vraies classes:",test_labels)
    
    pred = clf.predict(X_test)
    
    # Visualisation des résultats
    cm = confusion_matrix(test_labels, pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
    disp.plot()
    
    # Sauvegarde de la figure
    plt.savefig(f'../../Output/{args.vectorize}-{args.model}.png')

#__________MAIN
if __name__ == "__main__":
    main()

###_END_###