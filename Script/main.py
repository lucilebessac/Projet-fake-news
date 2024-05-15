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

import pandas as pd # Module pandas pour manipuler des tableurs
import matplotlib.pyplot as plt  # Module pour la création de graphiques et de visualisations
import nltk  # Bibliothèque pour le traitement du langage naturel
from nltk.corpus import stopwords  # Corpus de mots vides pour différentes langues
from nltk.stem import WordNetLemmatizer  # Outil pour la lemmatisation
from nltk.tokenize import word_tokenize  # Outil pour la tokenisation
from sklearn.ensemble import RandomForestClassifier  # Classe pour l'algorithme de RandomForest
from sklearn.feature_extraction.text import (  # Pour la vectorisation de texte
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.metrics import (  # Pour l'évaluation des modèles
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB  # Pour l'algorithme de Naive Bayes multinomial
from sklearn.svm import LinearSVC  # Pour l'algorithme SVM linéaire
from sklearn.tree import DecisionTreeClassifier  # Pour l'algorithme de l'arbre de décision

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
        help="Supprimer les stopwords de l'analyse.",
    )
    parser.add_argument(
        "-l",
        "--lemmatize",
        choices=["yes", "no"],
        help="Lemmatiser le texte.",
    )
    parser.add_argument(
        "-v",
        "--vectorize",
        choices=["count", "tfidf", "hash"],
        help="Choisir le vectorizer : 'count' pour CountVectorizer | 'tfidf' pour TfidfVectorizer | 'hash' pour HashingVectorizer",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["svc", "multi", "dtree", "rforest"],
        help="Choisir un modèle : 'svc' pour LinearSVC | 'multi' pour MultinomialNB | 'dtree' pour DecisionTree | 'rforest' pour RandomForest",
    )
    parser.add_argument(
        "-t",
        "--table",
        choices=["all"],
        help="Retourne un tableau avec la valeur de la precision en fonction du modèle",
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
    
    if args.preprocess and args.lemmatize and args.vectorize and args.model: 

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
        filename = f"../Output/{args.preprocess}_{args.lemmatize}_{args.vectorize}_{args.model}"
        plt.savefig(f"{filename}.png")
        
    elif args.table :
        preprocess_list = ["yes", "no"]
        lemmatize_list = ["yes", "no"]
        vectorizers_list = ["count", "tfidf", "hash"]
        models_list = ["svc", "multi", "dtree", "rforest"]
        dico_score = {"modèles":[], "précision":[]}
        exclude = ["yes_yes_hash_multi", "no_yes_hash_multi", "yes_no_hash_multi", "no_no_hash_multi"]
        
        for preprocess in preprocess_list:
            for lemmatize in lemmatize_list:
                for vertors in vectorizers_list:
                    for models in models_list:
                        element = f"{preprocess}_{lemmatize}_{vertors}_{models}"
    
                        if element not in exclude:
                            print(element)
                            train_txt = preprocess_text(train_txt, french_stop_words, lemmatize)
                            test_txt = preprocess_text(test_txt, french_stop_words, lemmatize)
                            
                            # Vectorisation du texte
                            if vertors == "count":
                                vectorizer = CountVectorizer()
                            elif vertors == "tfidf":
                                vectorizer = TfidfVectorizer()
                            elif vertors == "hash":
                                vectorizer = HashingVectorizer()
                        
                            X_train = vectorizer.fit_transform(
                                train_txt
                            )  # Ajustement sur les données d'entraînement
                            X_test = vectorizer.transform(
                                test_txt
                            )  # Transformation des données de test en utilisant le même vocabulaire
                        
                            # Entrainement du modèle sur le Train
                            if models == "svc":
                                clf = LinearSVC().fit(X_train, train_labels)
                            elif models == "multi":
                                clf = MultinomialNB().fit(X_train, train_labels)
                            elif models == "dtree":
                                clf = DecisionTreeClassifier().fit(X_train, train_labels)
                            elif models == "rforest":
                                clf = RandomForestClassifier().fit(X_train, train_labels)
    
                            #Precision
                            precision = round(clf.score(X_test, test_labels), 2)
                            print(precision)
                            
                            # Ajout du modèle et de sa précision dans le dictionnaire
                            dico_score["modèles"].append(element)
                            dico_score["précision"].append(precision)
        
        # Convertir le dictionnaire en DataFrame
        panda = pd.DataFrame(dico_score)

        # Trier les scores par ordre croissant
        sorted_panda = panda.sort_values(by="précision")

        # Réinitialiser les index pour avoir des index consécutifs
        sorted_panda.reset_index(drop=True, inplace=True)

        # Calculer la moyenne et l'écart-type des scores
        mean_score = sorted_panda['précision'].mean()
        std_score = sorted_panda['précision'].std()
        
        # Créer la figure et les axes avec une taille plus grande
        fig, ax = plt.subplots(figsize=(12, 8))

        # Créer un graphique avec une courbe
        plt.plot(sorted_panda['modèles'], sorted_panda['précision'], marker='o', linestyle='-')
        plt.xlabel('Modèles')
        plt.ylabel('Précision')
        plt.title('Précision par rapport aux modèles')

        # Afficher la moyenne et l'écart-type sur le graphique
        plt.text(1, 0, f"Moyenne: {mean_score:.2f}\nÉcart-type: {std_score:.2f}", transform=plt.gca().transAxes)
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Sauvegarde de la figure
        filename = f"../Rapports/score"
        plt.savefig(f"{filename}.png")
        
    else:
        print("Commande invalide, consulter le READ-ME ou taper la commande python3 main.py -h")
                        

# __________MAIN
# Appel de la fonction main
if __name__ == "__main__":
    main()

# _END