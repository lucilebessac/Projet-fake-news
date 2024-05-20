# Projet-fake-news

## Description

Dépôt Git pour le projet de groupe du cours d'extraction d'informations. Projet détection de fake news. 2023-2024.

Ce projet a pour but de comparer différentes méthodes de preprocessing, de vectorisation et de classification de fake news en français, et de déterminer la meilleure combinaison.

## Compte Rendu

Pour lire le compte rendu intégrale réalisé par l'équipe, rendez vous sur la branche Doc.

## Table des matières

- [Projet-fake-news](#projet-fake-news)
  - [Description](#description)
  - [Table des matières](#table-des-matières)
  - [Installation](#installation)
  - [Utilisation](#utilisation)
    - [Arguments positionnels :](#arguments-positionnels-)
    - [Arguments facultatifs :](#arguments-facultatifs-)
  - [Exemple](#exemple)
  - [Boîte à outils](#boîte-à-outils)


## Installation

Ce programme python est prévu pour fonctionner sous Linux.
Installez les librairies suivantes :

```bash
$ pip install pandas
$ pip install matplotlib
$ pip install nltk
$ pip install sklearn
$ pip install argparse
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

Le script principal est accompagné d'une boîte à outils composée des scripts `scrapping.py`, `enrichissement_corpus.py`, `modif_ids_data.py`.

- `scrapping.py` : sert à scrapper le web pour créer un fichier corpus au format .json, 
- `enrichissement_corpus.py` : sert à ajouter des textes (stockés en .txt) au corpus (au format .json),
- `modif_ids_data.py` : sert a  réarranger les id en cas de suppression d'un item.

