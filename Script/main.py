#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:22:22 2024

@author: guilhem, lucile, constance
"""

#__________MODULES
import argparse
import math
import random
import sys

import matplotlib.pyplot as plt
import nltk # Bibliothèque pour le traitement du langage naturel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from datastructures import get_index, load_json, save_json, split

nltk.download('wordnet')
random.seed(42)

#__________FUNCTIONS
def preprocess_text(text_list, stop_words, lemmatize):
    """ Cette fonction prétraite le texte en supprimant les stopwords et en lemmatisant si nécessaire """
    vectorizer = CountVectorizer(stop_words=stop_words, token_pattern=r'\b[^\d\W]+\b')
    vectors = vectorizer.fit_transform(text_list)
    preprocessed_text = [' '.join(vectorizer.inverse_transform(vector)[0]) for vector in vectors]
    if lemmatize == "yes":
        lemmatizer = WordNetLemmatizer()
        preprocessed_text = [' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text)]) for text in preprocessed_text]
    return preprocessed_text

def precision_rappel(pred, test_labels):
    classes = set(test_labels)
    compare = {}
    for i, element in enumerate(test_labels):
        if pred[i] == element:
            compare[element] = compare.get(element, 0) +1
    for classe in compare:
        print("Précision {classe}: {score}".format(
            classe=classe,
            score=compare[classe]/len([x for x in pred if x == classe]))
            )

    print("")
    for classe in compare:
        print("Rappel {classe}: {score}".format(
            classe=classe,
            score=compare[classe]/len([x for x in test_labels if x == classe])))


def main():
    # Fonction principale
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
        "-l",
        "--lemmatize",
        choices=["yes", "no"],
        default="no",
        help="Lemmatiser le texte."
    )
    parser.add_argument(
        "-v",
        "--vectorize",
        choices=["count", "tfidf", "hash"],
        default="count",
        help="Choisir le vectorizer : 'count' pour CountVectorizer | 'tfidf' pour TfidfVectorizer | 'hash' pour HashingVectorizer"
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["svc", "multi", "dtree", "rforest"],
        default="svc",
        help="Choisir un modèle : 'svc' pour LinearSVC | 'multi' pour MultinomialNB | 'dtree' pour DecisionTree | 'rforest' pour RandomForest"
    )
    args = parser.parse_args()
    # Vérification du format du fichier
    if args.file.split(".")[-1] != "json":
        print("Le fichier contenant les données doit être au format JSON")
        sys.exit()
    
    # Chargement du jeu de données
    dataset = load_json(args.file)
    
    # Découpage des données en 2 set : train(90%), test(10%)
    id_liste = [item["id"] for item in dataset]
    
    # Train
    size_train = math.ceil(len(id_liste) * 0.9)
    train = random.sample(id_liste, size_train)
    corpus_train = split(dataset, train, "../Corpus/train.json")

    # Test
    test = list(set(id_liste).difference(train))
    corpus_test = split(dataset, test, "../Corpus/test.json")
    
    # Récupération du contenu textuel et des labels "rating" pour chaque ensemble
    train_txt = corpus_train["content"]
    test_txt = corpus_test["content"]
    train_labels = corpus_train["rating"]
    test_labels = corpus_test["rating"]
    
    # Liste de stopwords en français et prétraitement du texte si demandé
    french_stop_words = stopwords.words('french')

    if args.preprocess == "yes":
        train_txt = preprocess_text(train_txt, french_stop_words, args.lemmatize)
        test_txt = preprocess_text(test_txt, french_stop_words, args.lemmatize)

    # Vectorisation du texte
    if args.vectorize == "count":
        vectorizer = CountVectorizer()
    elif args.vectorize == "tfidf":
        vectorizer = TfidfVectorizer()
    elif args.vectorize == "hash":
        vectorizer = HashingVectorizer()

    X_train = vectorizer.fit_transform(train_txt) # Ajustement sur les données d'entraînement
    X_test = vectorizer.transform(test_txt) # Transformation des données de test en utilisant le même vocabulaire

    # Entrainement du modèle sur le Train
    if args.model == "svc":
        clf = LinearSVC().fit(X_train, train_labels)
    elif args.model == "multi":
        clf = MultinomialNB().fit(X_train, train_labels)
    elif args.model == "dtree":
        clf = DecisionTreeClassifier().fit(X_train, train_labels)
    elif args.model == "rforest":
        clf = RandomForestClassifier().fit(X_train, train_labels)

    
    #Prédictions
    pred = clf.predict(X_test)

    # Classification Report
    report = classification_report(test_labels, pred)
    print("Classification Report:\n", report)

    # Visualisation des résultats
    cm = confusion_matrix(test_labels, pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
    disp.plot()
    
    # Sauvegarde de la figure
    filename = f'../Output/{args.vectorize}-{args.model}'
    if args.preprocess == 'yes':
        filename += '-preprocessed'
    if args.lemmatize == 'yes':
        filename += '-lem'
    plt.savefig(f'{filename}.png')
    
#__________MAIN
if __name__ == "__main__":
    main()

###_END_###