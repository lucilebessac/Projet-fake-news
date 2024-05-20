#__________MODULES
import pandas as pd
import requests
import json
from bs4 import BeautifulSoup as soup
from typing import List, Dict

#__________FUNCTIONS
def get_html(url: str, attribut: str, search_for: str) -> soup:
    """
    Récupère le contenu HTML d'une URL et trouve un élément spécifique selon un attribut.

    Parameters:
        url (str): L'URL à partir de laquelle récupérer le contenu HTML.
        attribut (str): L'attribut à utiliser pour la recherche ("id" ou "class").
        search_for (str): La valeur de l'attribut à rechercher.

    Returns:
        BeautifulSoup: L'objet BeautifulSoup contenant le résultat de la recherche.
    """
    html = requests.get(url)
    # print(html)
    html.encoding = 'utf-8'
    page = soup(html.text, 'html.parser')
    # print(page)
    if attribut == "id":
        variable = page.find("div", id=search_for)
    elif attribut == "class":
        variable = page.find("div", class_=search_for)
    # print(variable)
    return variable

def save_html(url: str, liste_href: List[dict]) -> List[str]:
    """
    Télécharge le contenu HTML des liens fournis dans une liste et les enregistre en tant que fichiers HTML.

    Parameters:
        url (str): L'URL de base pour les liens.
        liste_href (List[dict]): La liste des liens représentés sous forme de dictionnaires.

    Returns:
        List[str]: La liste des URL complètes des fichiers HTML enregistrés.
    """
    liste_url = []
    for href in liste_href:
        link_url = href["href"]
        full_url = url + link_url
        # print(full_url)
        liste_url.append(full_url)
        html = requests.get(full_url)
        # print(html)
        liste = full_url.split("/")
        # print(liste)
        if liste[-1] == "":
            name_file = "index"
        else:
            name_file = liste[-1]
        # print(name_file)
        with open(f"../Data/RawData/{name_file}.html", "w", encoding="utf8") as file:
            file.write(html.text)
    return liste_url

def get_article(url: str, source: str, urls: list) -> pd :
    """
    Récupère les articles à partir des URLs fournies et retourne un objet Corpus.

    Parameters:
        url (str): L'URL de base pour la construction des URLs des articles.
        source (str): La source des articles.
        urls (List[str]): La liste des URLs des pages d'où extraire les articles.

    Returns:
        Corpus: L'objet Corpus contenant les articles extraits.
    """
    data = {"url": [], "author": [], "date": [], "label": [], "title": [], "resume": [], "content": [], "category": []}

    liste_url = []
    for link in urls:
        print(link)
        cat = link.split("/")
        # print(liste)
        category = cat[-1]
        main = get_html(link, "id", "main")
        articles = main.find_all("div", class_="row")
        for article in articles:
            #print(article)
            note_article = article.find("span")
            # print(note_article)
            infos_article = article.find("div", class_="info_article")
            # print(infos_article)
            url_article = article.find("a")
            # print(url_article)
            if url_article and note_article and infos_article:
                # print(article)
                href = url_article["href"]
                # print(href)
                url_final = url + href
                # print(url_final)
                if url_final not in urls and url_final not in liste_url :
                    note = note_article.text.strip()
                    # print(note)
                    infos = infos_article.text.strip().split()
                    date = infos[0]
                    # print(date)
                    auteur = infos[2]
                    # print(auteur)
                    item = requests.get(url_final)
                    content = soup(item.text, 'html.parser')
                    # print(content)
                    header = content.find_all("title")
                    titre = header[0].text.strip()
                    titre = titre.replace("- Hoaxbuster", "").strip()
                    resume = header[-1].text.strip()
                    resume = resume.replace("- Hoaxbuster", "").strip()
                    # print(titre)
                    texts = " ".join([text.text.strip() for text in content.find_all("p")])
                    data["url"] += [url_final]
                    data["author"] += [auteur]
                    data["date"] += [date]
                    data["label"] += [note]
                    data["title"] += [titre]
                    data["resume"] += [resume]
                    data["content"] += [texts]
                    data["category"] += [category]
                    liste_url.append(url_final)
    dataframe = pd.DataFrame(data)
    return dataframe

def save_json(dataframe: pd, filename: str) -> None:
    """
    Enregistre un objet pandas au format JSON.

    Parameters:
        dataframe (pandas.DataFrame): Le DataFrame à enregistrer.
        filename (str): Le nom du fichier JSON.

    Returns:
        None
    """
    with open(filename, 'w', encoding='utf-8') as file:
        dataframe.to_json(file, orient='records', indent=4, force_ascii=False)


def main():
    """
    Fonction principale pour l'exécution du programme.
    """
    url = "https://www.hoaxbuster.com"
    navbar = get_html(url, "id", "navbar")
    # print(navbar)

    links = navbar.find_all("a")
    links.pop(0)
    # print(links)

    urls = save_html(url, links)
    # print(urls)

    source = url.split(".")
    source = source[-2]
    # print(source)

    spreadsheet = get_article(url, source, urls)
    # print(spreadsheet)

    # Enregistrer le DataFrame au format JSON
    save_json(spreadsheet, "../Data/Dataset/data.json")
    #save_json(corpus)

#__________MAIN

if __name__ == "__main__":
    main()

###_END