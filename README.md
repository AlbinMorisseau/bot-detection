# Détection de Bots à partir de Logs HTTP avec XGBoost

Ce projet a pour objectif de construire un modèle de machine learning capable de différencier le trafic généré par des humains de celui généré par des bots, en se basant sur l'analyse de logs du moteur de recherche de la bibliothèque et centre d’information de l’Université Aristote de Thessalonique, en Grèce. (https://zenodo.org/records/3477932)

Les journaux de serveur obtenus couvrent un mois complet, du 1er au 31 mars 2018, et comprennent 4 091 155 requêtes, avec une moyenne de 131 973 requêtes par jour et un écart-type de 36 996,7 requêtes. Au total, les requêtes proviennent de 27 061 adresses IP uniques et de 3 441 chaînes user-agent distinctes.

J'ai utilisé une version traitée des journaux de logs, sous forme d’un dataset étiqueté, où les entrées sont regroupées en sessions et accompagnées de leurs caractéristiques extraites.

## Table des Matières
1. [Contexte du Projet](#contexte-du-projet)
2. [Structure du Projet](#structure-du-projet)
3. [Installation](#installation)
4. [Workflow](#workflow)
5. [Analyse Exploratoire (EDA)](#analyse-exploratoire-eda)
6. [Modélisation](#modélisation)
7. [Résultats](#résultats)
8. [Conclusion et Pistes d'Amélioration](#conclusion-et-pistes-damélioration)

## Contexte du Projet

Les bots malveillants représentent une menace croissante pour les infrastructures web, notamment dans les secteurs du e-commerce et du luxe. Selon le Global Bot Security Report 2024 de l'entreprise DataDome, 95 % des bots avancés passent inaperçus, ce qui expose les sites à des risques tels que le vol de contenu, le scraping de données sensibles et des attaques par déni de service distribué (DDoS) 
De plus, 60 % des sites n'ont mis en place aucune protection contre les attaques basiques de bots.

Dans ce contexte, l'objectif de ce projet est de développer des méthodes d'analyse permettant d'identifier et de différencier les comportements des utilisateurs humains de ceux des robots. Une telle approche est essentielle pour protéger les ressources informatiques, préserver l'intégrité des données et optimiser l'expérience utilisateur en filtrant les interactions non humaines.

## Structure du Projet
'''
bot-detection-project/
├── data/
│   └── data.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── preprocess.py
│   └── train.py
├── .gitignore
├── requirements.txt
└── README.md
'''          