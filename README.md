# Explorateur de texte

Ce projet a pour but de créer une interface d'analyse de texte permettant d'avoir une vision à la fois d'ensemble et détaillée de plusieurs documents.
L'objectif est de permettre la comparaison entre plusieurs sources et l'observation des liens entre de multiples sujets.


Le projet est basé sur 3 technologies différentes :
- Spacy qui permet d'extraire depuis le texte des informations pertinentes.
- ArangoDB qui permet de stocker sous forme de graphe ces informations.
- Plotly-Dash pour la construction de l'interface.

Quatres visualisations sont disponibles :
- Un graphe de co-occurence entre les termes les plus employés.
- Les termes grammaticalements associés à un mot en particulier et les documents porteurs de ces associations.
- Un concordancier permettant de lire les passages porteurs d'intérêt.
- Une barre de recherche avec calcul de similarité pour directement trouver des phrases. 

L'outil cherche à faciliter des tâches comme la comparaison d'articles de journaux, la veille scientifique et la citation de sources. 
