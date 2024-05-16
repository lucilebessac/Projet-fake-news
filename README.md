# Projet-fake-news

## Description

Dépôt Git pour le projet de groupe du cours d'extraction d'informations. Projet détection de fake news. 2023-2024.

Ce projet a pour but de comparer différentes méthodes de preprocessing, de vectorisation et de classification de fake news en français, et de déterminer la meilleure combinaison.

## Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Exemple](#exemple)
- [Boîte à outils](#boîte_à_outils)


## Installation

Ce programme python est prévu pour fonctionner sous Linux.
Installez les librairies suivantes :

```bash
$ pip install matplotlib
$ pip install nltk
$ pip install sklearn
$ pip install datastructures
```

## Utilisation

Pour utiliser ce programme, exécutez `main.py` avec les arguments suivants :

```bash
$ python main.py [-p {yes,no}] [-l {yes,no}] [-v {count,tfidf,hash}] [-m {svc,multi,dtree,rforest}] file
```

### Arguments positionnels :

`file` : Chemin vers le fichier qui contient le corpus au format JSON.

### Arguments facultatifs :

`-h`, `--help` : Affiche le message d'aide et quitte le programme.

`-p` `{yes,no}`, `--preprocess` `{yes,no}` : Supprime les stopwords de l'analyse.

`-l` `{yes,no}`, `--lemmatize` `{yes,no}` : Lemmatise le texte.

`-v` `{count,tfidf,hash}`, `--vectorize` `{count,tfidf,hash}` : Choisissez le vectorizer :
        'count' pour CountVectorizer
        'tfidf' pour TfidfVectorizer
        'hash' pour HashingVectorizer

`-m` `{svc,multi,dtree,rforest}`, `--model` `{svc,multi,dtree,rforest}` : Choisissez un modèle :
        'svc' pour LinearSVC
        'multi' pour MultinomialNB
        'dtree' pour DecisionTree
        'rforest' pour RandomForest

`-t`, `--table` : Retourne un graphe comparant la valeur de la precision de chaque modèle.


## Exemple

```bash
$ python main.py -p yes -l yes -v tfidf -m svc -t ../Data/data_enrichi.json
```

## Boîte à outils

Le script principal est accompagné d'une boîte à outils composée des scripts `enrichissement_corpus.py` et `modif_ids_data.py`, servant respectivement à ajouter de nouveaux textes au format .txt au corpus au format .json, et à réarranger les id en cas de suppression de d'un item.

