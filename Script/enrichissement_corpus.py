import os
import json

"""
Ce script charge des articles dans des fichiers .txt à partir d'un dossier spécifié, extrait les titres et les contenus de ces articles,
puis les ajoute à un fichier JSON existant. Il utilise également le dernier ID du fichier JSON pour attribuer des IDs
uniques aux nouveaux éléments ajoutés.

dossier_nouveaux_articles : dossier/*.txt - le dossier contenant les fichier txt à ajouter
fichier_corpus : .json - le fichier d'entrée à enrichir
fichier_corpus_enrichi : .json - le fichier de sorti, peut être le même que le fichier d'entrée
Auteur : Lucile BESSAC
Date : 04/05/2024

"""

# Charge le document JSON pour récupérer le dernier ID
def get_dernier_id(corpus_json_existant):
    with open(corpus_json_existant) as f:
        corpus_json_existant = json.load(f)

    # Récupérer la liste des items
    items = corpus_json_existant['items']

    # Trouver le dernier ID
    dernier_id = max(item['id'] for item in items)

    return dernier_id

# Récupère le titre et le contenu de l'article et l'ajoute au json
def txt_to_json(dossier_nouveaux_articles, fichier_corpus, fichier_corpus_enrichi):
    nouveaux_items_articles = []
    nouvel_id = get_dernier_id(fichier_corpus)
    for filename in os.listdir(dossier_nouveaux_articles):
        if filename.endswith(".txt"):
            with open(os.path.join(dossier_nouveaux_articles, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                # Assurez-vous qu'il y a au moins une ligne dans le fichier
                if lines:
                    title = lines[0].strip()  # Première ligne pour le titre
                    content = ''.join(lines[1:])  # Le reste pour le contenu
                    nouvel_id += 1

                    # Création de l'élément article JSON
                    item = {
                        "id": nouvel_id,
                        "rating": "Vrai",
                        "title": title,
                        "content": content
                    }
                    nouveaux_items_articles.append(item)

    with open(fichier_corpus, 'r+', encoding='utf-8') as input_file:
        data = json.load(input_file)  # Chargement des données existantes
        data['items'].extend(nouveaux_items_articles)  # Ajout des nouveaux articles

    with open(fichier_corpus_enrichi, 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)

# Appel de la fonction principale
if __name__ == "__main__":
    dossier_nouveaux_articles = "../../../Corpus 2022/dev"
    fichier_corpus = "../Data/data_enrichi.json"
    fichier_corpus_enrichi = "../Data/data_enrichi.json"
    txt_to_json(dossier_nouveaux_articles, fichier_corpus, fichier_corpus_enrichi)
