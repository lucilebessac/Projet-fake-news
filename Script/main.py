#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 21:22:22 2024

@author: guilhem
"""

#__________MODULES
import json
import pandas as pd
import random
random.seed(42)
import math

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from dataclasses import dataclass, asdict, field
from typing import List, Dict
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

#__________DATACLASS
@dataclass
class Article:
    id: int
    url: str
    author: str
    date: str
    rating: str
    title: str
    resume: str
    content: str
    category: str

@dataclass
class Corpus:
    items: list[Article]

#__________FUNCTIONS
def save_json(corpus: Corpus, path: str) -> None:
    """
    Sauvegarde un objet Corpus au format JSON.

    Parameters:
    corpus (Corpus): L'objet Corpus à sauvegarder.

    Returns:
        None
    """
    data = {"items": []}
    for item in corpus.items:
        item_dict = asdict(item)
        data["items"].append(item_dict)
    with open(path, "w", encoding="utf8") as file:
        json.dump(data, file, indent=2)


def load_json(path_corpura: str) -> Dataset : 
    """
    Charge un fichier JSON et le convertit en objet Dataset.

    Parameters :
    path_corpura -- le chemin vers le fichier JSON à charger

    Returns :
    Un objet Dataset contenant les données du fichier JSON
    """
    with open(path_corpura, "r", encoding="utf8") as file:
        json_data = json.load(file)
    corpus = Corpus([Article(**item) for item in json_data['items']])
    data = pd.DataFrame([vars(article) for article in corpus.items])
    data = Dataset.from_pandas(data)
    return data

def get_index(dataset: List[Dict[str, str]], name: str) -> Dict[str, List[int]]:
    """
    Récupère des informations sur les éléments du dataset en fonction de leur note.

    Parameters :
    dataset -- une liste de dictionnaires représentant les données
    name -- une chaîne de caractères représentant le nom de l'ensemble de données

    Returns :
    Un dictionnaire contenant des listes d'identifiants d'éléments, classées par note
    """
    
    liste_true = []
    liste_false = []
    liste_mix = []
    liste_unknow = []
    
    for item in dataset:
        if item["rating"] == "Vrai":
            liste_true.append(item["id"])
        elif item["rating"] == "Faux":
            liste_false.append(item["id"])
        elif item["rating"] == "Du vrai / du faux":
            liste_mix.append(item["id"])
        else: 
            liste_unknow.append(item["id"])
            
    dico_info = {"Vrai" : liste_true,
                 "Faux" : liste_false,
                 "Mix" : liste_mix,
                 "Inconnue" : liste_unknow
                 }
    
    nb_true = len(liste_true)
    nb_false = len(liste_false)
    nb_mix = len(liste_mix)
    nb_unknow = len(liste_unknow)
            
    print(f"Infos corpus {name}: {nb_true=}, {nb_false=}, {nb_mix=}, {nb_unknow=}")
    return dico_info

def split(dataset: List[Dict[str, str]], data: List[int], path: str) -> List[Article] :
    """
    Divise un dataset en un sous-ensemble en fonction d'une liste d'indices donnée, puis sauvegarde ce sous-ensemble au format JSON.
    
    Parameters :
    dataset -- une liste de dictionnaires représentant les données complètes
    data -- une liste d'entiers représentant les indices des éléments à extraire
    path -- le chemin vers le fichier JSON de sortie
    
    Returns :
    Une liste d'objets Article correspondant au sous-ensemble extrait
    """
    matched = []
    for item in data:        
        element = dataset[item]
        # print(element)
        match = Article(id=element["id"], 
                        url=element["url"], 
                        author=element["author"], 
                        date=element["date"], 
                        rating=element["rating"], 
                        title=element["title"], 
                        resume=element["resume"], 
                        content=element["content"], 
                        category=element["category"]
                        )
        matched.append(match)
    split_corpus = Corpus(matched)
    save_json(split_corpus, path)
    
    return load_json(path)

def main():
    path_corpura = "../Data/data_enrichi.json"
    
    # Appel la fonction load_json
    dataset = load_json(path_corpura)
    
    corpus_index = get_index(dataset, "origin")
    # print(corpus_index)
    
    # Découpage des données en 3 set : train(80%), test(10%), dev(10%)
    id_liste = [item["id"] for item in dataset]
    
    # Train
    size_train = math.ceil(len(id_liste) * 0.8)
    train = random.sample(id_liste, size_train)
    corpus_train = split(dataset, train, "../Corpus/train.json")
    train_index = get_index(corpus_train, "train")

    # Test
    remain = list(set(id_liste).difference(train))
    test = random.sample(remain, math.ceil((len(id_liste) - len(train)) / 2))
    corpus_test = split(dataset, test, "../Corpus/test.json")
    test_index = get_index(corpus_test, "test")
    
    # Dev
    dev = list(set(remain).difference(test))
    corpus_dev = split(dataset, dev, "../Corpus/dev.json")
    dev_index = get_index(corpus_dev, "dev")
    
    # On recupère le contenu textuel de chaque set
    train_txt = corpus_train["content"]
    test_txt = corpus_test["content"]
    dev_txt = corpus_dev["content"]
    
    # On vectorise 
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_txt) # On utilise la methode .fit_tranform() pour le train
    X_test = vectorizer.transform(test_txt) # On utilise la methode .tranform() pour le test
    X_dev =  vectorizer.transform(dev_txt)
    
    # On recupère le label "rating" de chaque set
    train_labels = corpus_train["rating"]
    test_labels = corpus_test["rating"]
    dev_labels = corpus_dev["rating"]
    
    # Entrainement du modèle sur le Train
    clf = LinearSVC().fit(X_train, train_labels)

    #Score
    print(clf.score(X_test, test_labels))
    
    #Predict
    print("predictions:", clf.predict(X_test))
    
    #Donnee
    print("vraies classes:",test_labels)
    
    pred = clf.predict(X_test)
    
    # Visualisation des résultats
    cm = confusion_matrix(test_labels, pred, labels=clf.classes_)
    ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()


#__________MAIN
if __name__ == "__main__":
    main()

###_END_###