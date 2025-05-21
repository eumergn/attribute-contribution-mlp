# Projet IA – Contribution des Attributs dans un Réseau de Neurones

Ce projet a pour objectif d'expliquer les prédictions d’un réseau de neurones (MLP) en analysant la contribution locale des attributs, en s'inspirant de la méthode LIME.

##  Structure du projet

```
.
├── main.py                        # Script principal (entraînement, test, explication)
├── data/
│   └── iris_extended.csv         # Jeu de données (version étendue d'Iris)
├── results/
│   ├── models/                   # Modèle MLP sauvegardé
│   └── visualisations/           # Graphiques et analyses locales
└── src/
    ├── train.py                  # Entraînement du réseau de neurones
    ├── evaluate.py              # Évaluation du modèle (précision, matrice de confusion)
    ├── explain_local.py         # Perturbations + modèle explicatif local
    └── pre_process.py           # Chargement et transformation des données
```


##  Exécution

Lancez simplement :
```bash
python3 main.py
```

Les résultats (modèle, courbes d’apprentissage, contributions locales) sont sauvegardés dans `results/`.

##  Fonctionnalités

- Entraînement d’un MLP sur un jeu multiclasse (Iris étendu).
- Sélection d’instances à expliquer (correctes + incorrectes).
- Génération de perturbations autour d’une instance.
- Entraînement local d’un modèle linéaire explicatif.
- Visualisation des attributs les plus influents.
- Comparaison de la stabilité des explications en variant l’amplitude du bruit (±10 % vs ±20 %).

##  Rapport

Le rapport PDF (présent dans ce dépôt) contient :
- Une description du modèle et des données
- Les résultats détaillés
- L’analyse locale et graphique des attributs
- Une discussion sur les limites et perspectives

##  Auteurs

Projet réalisé par :
- Armagan Omer  
- Khadjiev Djakar  
- Parfeni Alexandru  

Université de Strasbourg – L3 Informatique – 2024/2025

##  Outils utilisés

- Python 3
- PyTorch
- scikit-learn
- matplotlib
- Overleaf (pour la rédaction du rapport en LaTeX)
