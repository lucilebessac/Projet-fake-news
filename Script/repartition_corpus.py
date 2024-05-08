import json

def count_items_by_rating(json_file):
    # Dictionnaire pour stocker le nombre d'items pour chaque rating
    rating_count = {}

    # Chargement du fichier JSON
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Parcours de tous les items
    for item in data['items']:
        rating = item.get('rating', 'Non évalué')  # Récupération du rating de l'item
        # Incrémentation du compteur pour ce rating
        rating_count[rating] = rating_count.get(rating, 0) + 1

    return rating_count

# Exemple d'utilisation
if __name__ == "__main__":
    json_file = "../Data/data_enrichi.json"
    rating_count = count_items_by_rating(json_file)
    for rating, count in rating_count.items():
        print(f"Rating: {rating}, Nombre d'items: {count}")
