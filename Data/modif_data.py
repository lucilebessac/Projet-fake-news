import json

# Charger le fichier JSON
with open('data_enrichi.json', 'r') as f:
    data = json.load(f)

# Modifier les identifiants
for i, item in enumerate(data['items'], start=0):
    item['id'] = i

# Enregistrer le fichier modifié
with open('data_enrichi_modifie.json', 'w') as f:
    json.dump(data, f, indent=4)
