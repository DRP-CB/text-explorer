# Explorateur de texte

Ce projet a pour but de créer une interface d'analyse de texte permettant d'avoir une vision à la fois d'ensemble et détaillée de plusieurs documents.
L'objectif est de permettre la comparaison entre plusieurs sources et l'observation des liens entre de multiples sujets.

Le projet est basé sur 3 technologies différentes :
- Spacy qui permet d'extraire depuis le texte des informations pertinentes.
- ArangoDB qui permet de stocker sous forme de graphe ces informations.
- Plotly-Dash pour la construction de l'interface.

Trois visualisations sont envisagées :
- Un graphe de co-occurence entre les termes les plus employés.
- Les termes les plus associés à un mot en particulier et les sources de ces associations.
- Un concordancier permettant de voir dans le texte directement les passages porteurs d'intérêt.

L'outil a pour but de faciliter des tâches comme la comparaison d'articles de journaux et la veille scientifique. 