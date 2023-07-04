# Projet Super Hedge d'Options Européennes

Ce repo est dédié au projet de sureplication (Super Hedge en anglais) d'options européennes. Plus précisément, il concerne le pricing d'une option européenne sous condition de sureplication parfaite sur toute la durée du backtest (de 2000 à 2023).

## Objectif

L'objectif principal du projet est de concevoir et de mettre en œuvre un modèle basé sur la théorie des supports de prix. Vous pouvez trouver l'article de recherche pertinent en suivant le lien ci-dessous :
- [Lien vers l'article de recherche](https://hal.science/hal-02379707/document)

Notre approche repose sur l'utilisation de l'historique des prix open, close, high, low pour prédire les fluctuations journalières. L'optimisation et la calibration des paramètres du modèle sont effectuées sous deux contraintes :

1. Maximiser le pourcentage de suréplication, défini comme `(jours surepliqués / n jours total) * (la différence entre la valeur finale du portefeuille et le payoff)`.
2. Minimiser le prix de vente de l'option, c'est-à-dire la valeur initiale du portefeuille (V0).

## Ressources

Le notebook Colab final pour le projet peut être consulté ici :
- [Lien vers le notebook Colab](https://github.com/ElAm1ne/March--Fi-2/blob/master/SuperHedge.ipynb)

## Technologies

Nous avons choisi d'utiliser Python en combinaison avec la librairie Numba pour ce projet. Cette combinaison permet des temps de calculs proches de ceux du langage C/C++, tout en conservant la lisibilité du code Python.

## Conclusion

Nous espérons que vous trouverez ce projet instructif et intéressant. Bonne lecture !
