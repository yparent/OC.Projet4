# Analyse des causes d'attrition — TechNova Partners

Projet réalisé dans le cadre du parcours Data Scientist d'OpenClassrooms.

## Contexte

TechNova Partners est une ESN spécialisée en transformation digitale. Le département RH fait face à un taux de turnover inhabituellement élevé et souhaite identifier les causes de départ des employés pour mettre en place des actions correctives.

On dispose de 3 fichiers de données internes (SIRH, évaluations de performance, sondage bien-être) portant sur environ 1 470 employés.

## Objectifs

- Explorer et nettoyer les données, puis les consolider en un seul jeu de données
- Construire un modèle de classification pour prédire le risque de départ
- Interpréter les résultats avec SHAP pour identifier les facteurs de démission
- Formuler des recommandations à destination des RH

## Structure du projet

```
├── data/
│   ├── extrait_sirh.csv
│   ├── extrait_eval.csv
│   └── extrait_sondage.csv
├── notebook_analyse_attrition.ipynb
├── presentation_soutenance.pptx
├── pyproject.toml
└── README.md
```

## Démarche

1. **Exploration** — Chargement des 3 fichiers, jointure sur l'identifiant employé, statistiques descriptives et visualisations comparatives (partants vs restants)
2. **Préparation** — Encodage des variables catégorielles, matrice de corrélation, feature engineering (ratio salaire/expérience, satisfaction moyenne, etc.)
3. **Modélisation** — Entraînement progressif : Dummy (baseline) → Régression Logistique → Random Forest → XGBoost. Gestion du déséquilibre de classes via SMOTE.
4. **Optimisation** — GridSearchCV sur XGBoost, détermination du seuil de probabilité optimal via la courbe precision-recall
5. **Interprétation** — SHAP (Beeswarm plot, Waterfall plots, scatter plots) et Permutation Importance pour identifier les causes d'attrition

## Résultats principaux

Les facteurs les plus déterminants dans le départ des employés sont :
- Les heures supplémentaires régulières
- Un revenu mensuel faible
- Un profil jeune avec peu d'ancienneté
- Une satisfaction basse (environnement de travail, équilibre pro/perso)

## Installation

```bash
# Avec uv (recommandé)
uv sync

# Ou avec pip
pip install -r requirements.txt
```

Le fichier `pyproject.toml` contient toutes les dépendances nécessaires.

### Prérequis

- Python 3.10 à 3.12
- Les packages listés dans `pyproject.toml` (pandas, scikit-learn, xgboost, imbalanced-learn, shap, etc.)

## Lancement

```bash
uv run jupyter notebook notebook_analyse_attrition.ipynb
```

## Auteur

Yohan Parent — Parcours Data Scientist, OpenClassrooms
