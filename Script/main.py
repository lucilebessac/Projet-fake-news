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
from dataclasses import dataclass, asdict, field
from typing import List, Dict
from datasets import Dataset

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

    Args:
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

    Arguments :
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

def infos(dataset: List[Dict[str, str]]) -> Dict[str, List[int]]:
    """
    Récupère des informations sur les éléments du dataset en fonction de leur note.

    Arguments :
    dataset -- une liste de dictionnaires représentant les données

    Returns :
    Un dictionnaire contenant des listes d'identifiants d'éléments, classées par note
    """
    liste_true = []
    liste_false = []
    liste_else = []
    liste_unknow = []
    
    for item in dataset:
        if item["rating"] == "Vrai":
            liste_true.append(item["id"])
        elif item["rating"] == "Faux":
            liste_false.append(item["id"])
        elif item["rating"] == "Du vrai / du faux":
            liste_else.append(item["id"])
        else: 
            liste_unknow.append(item["id"])
            
    dico_info = {"Vrai" : liste_true,
                 "Faux" : liste_false,
                 "Mix" : liste_else,
                 "Inconnue" : liste_unknow
                 }
    
    nb_true = len(liste_true)
    nb_false = len(liste_false)
    nb_else = len(liste_else)
    nb_unknow = len(liste_unknow)
            
    print(f"Infos corpus : {nb_true=}, {nb_false=}, {nb_else=}, {nb_unknow=}")
    return dico_info

def split (dataset: List[Dict[str, str]], data: List[int], path: str) -> List[Article] :
    """
    Divise un dataset en un sous-ensemble en fonction d'une liste d'indices donnée, puis sauvegarde ce sous-ensemble au format JSON.
    
    Arguments :
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
    
    return matched

def main():
    path_corpura = "../Data/data.json"
    
    # Appel la fonction load_json
    dataset = load_json(path_corpura)
    
    info = infos(dataset)
    # print(info)
    
    # Découpage des données en 3 set : train(80%), test(10%), dev(10%)
    id_liste = [item["id"] for item in dataset]
    
    # Train
    size_train = int(len(id_liste) * 0.8)  
    train = random.sample(id_liste, size_train)
    corpus_train = split(dataset, train, "../Corpus/train.json")
    
    # Test
    remain = list(set(id_liste).difference(train))
    test = random.sample(remain, math.ceil((len(id_liste) - len(train)) / 2))
    corpus_test = split(dataset, test, "../Corpus/test.json")
    
    # Dev
    dev = list(set(remain).difference(test))
    corpus_dev = split(dataset, dev, "../Corpus/dev.json")


#__________MAIN
if __name__ == "__main__":
    main()

###_END_###