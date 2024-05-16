#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 12:45:43 2024

@author: lucile

Decription : Ce script charge un fichier .json et réattribue des ids à tous les items du fichier,
afin de s'assurer que les id sont dans l'ordre croissant et qu'aucun id ne manque.

fichier_source : dossier/fichier.json - le dossier contenant le fichier .json duquel modifier les id
fichier_cible : dossier/fichier.json - le dossier contenant le fichier .json où sotkcer le résultat
"""

import json

fichier_source = '../Data/data_enrichi.json'
fichier_cible = '../Data/data_enrichi_modifié.json'


# Charger le fichier JSON
with open(fichier_source, 'r') as f:
    data = json.load(f)

# Modifier les identifiants
for i, item in enumerate(data['items'], start=0):
    item['id'] = i

# Enregistrer le fichier modifié
with open(fichier_cible, 'w') as f:
    json.dump(data, f, indent=4)
