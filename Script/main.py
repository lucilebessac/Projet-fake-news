#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:22:22 2024

@author: guilhem, lucile, constance
"""

#__________MODULES
import argparse  # Module pour analyser les arguments de la ligne de commande
import math  # Module pour les opérations mathématiques
import random  # Module pour la génération de nombres aléatoires
import sys  # Module fournissant un accès à certaines variables utilisées ou maintenues par l'interpréteur Python

import matplotlib.pyplot as plt  # Module pour la création de graphiques et de visualisations
import nltk  # Bibliothèque pour le traitement du langage naturel
from nltk.corpus import stopwords  # Corpus de mots vides pour différentes langues
from sklearn.ensemble import RandomForestClassifier  # Classe pour l'algorithme de RandomForest
from sklearn.feature_extraction.text import (  # Classes pour la vectorisation de texte
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import (  # Fonctions et classes pour l'évaluation des modèles
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB  # Classe pour l'algorithme de Naive Bayes multinomial
from sklearn.svm import LinearSVC  # Classe pour l'algorithme SVM linéaire
from sklearn.tree import DecisionTreeClassifier  # Classe pour l'algorithme de l'arbre de décision

from datastructures import (  # Importer les fonctions personnalisées depuis le fichier datastructures.py
    get_index,
    load_json,
    save_json,
    split,
    preprocess_text,
)

nltk.download("wordnet")  # Téléchargement des données WordNet nécessaires pour la lemmatisation
random.seed(42)  # Initialisation de la graine pour la reproductibilité des résultats aléatoires



#__________FUNCTION
def main():
    """
    Fonction principale qui teste différentes formes de vectorisation et de modèles pour la détection de fake-news.
    """
    parser = argparse.ArgumentParser(
        description="Ce programme a pour but de tester différentes formes de vectorisation et de models pour la détection de fake-news"
    )
    parser.add_argument(
        "file", help="Chemin vers le fichier qui contient le corpus au format json."
    )
    parser.add_argument(
        "-p",
        "--preprocess",
        choices=["yes", "no"],
        default="no",
        help="Supprimer les stopwords de l'analyse.",
    )
    parser.add_argument(
        "-l",
        "--lemmatize",
        choices=["yes", "no"],
        default="no",
        help="Lemmatiser le texte.",
    )
    parser.add_argument(
        "-v",
        "--vectorize",
        choices=["count", "tfidf", "hash"],
        default="count",
        help="Choisir le vectorizer : 'count' pour CountVectorizer | 'tfidf' pour TfidfVectorizer | 'hash' pour HashingVectorizer",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["svc", "multi", "dtree", "rforest"],
        default="svc",
        help="Choisir un modèle : 'svc' pour LinearSVC | 'multi' pour MultinomialNB | 'dtree' pour DecisionTree | 'rforest' pour RandomForest",
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
    size_index = len(id_liste)
    print(f"{size_index=}")

    # Set Train
    size_train = math.ceil(len(id_liste) * 0.9)
    train = random.sample(id_liste, size_train)
    corpus_train = split(dataset, train, "../Corpus/train.json")

    # Set Test
    test = list(set(id_liste).difference(train))
    size_test = len(test)
    corpus_test = split(dataset, test, "../Corpus/test.json")

    # Récupération du contenu textuel et des labels "rating" pour chaque ensemble
    train_txt = corpus_train["content"]
    test_txt = corpus_test["content"]
    train_labels = corpus_train["rating"]
    test_labels = corpus_test["rating"]

    # Liste de stopwords en français et prétraitement du texte si demandé
    french_stop_words = stopwords.words("french")

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

    X_train = vectorizer.fit_transform(
        train_txt
    )  # Ajustement sur les données d'entraînement
    X_test = vectorizer.transform(
        test_txt
    )  # Transformation des données de test en utilisant le même vocabulaire

    # Entrainement du modèle sur le Train
    if args.model == "svc":
        clf = LinearSVC().fit(X_train, train_labels)
    elif args.model == "multi":
        clf = MultinomialNB().fit(X_train, train_labels)
    elif args.model == "dtree":
        clf = DecisionTreeClassifier().fit(X_train, train_labels)
    elif args.model == "rforest":
        clf = RandomForestClassifier().fit(X_train, train_labels)

    # Prédictions
    pred = clf.predict(X_test)

    # Classification Report
    report = classification_report(test_labels, pred)
    print("Classification Report:\n", report)

    # Visualisation des résultats
    cm = confusion_matrix(test_labels, pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
    disp.plot()

    # Sauvegarde de la figure
    filename = f"../Output/{args.vectorize}-{args.model}"
    if args.preprocess == "yes":
        filename += "-preprocessed"
    if args.lemmatize == "yes":
        filename += "-lem"
    plt.savefig(f"{filename}.png")


# __________MAIN
# Appel de la fonction main
if __name__ == "__main__":
    main()

# _END