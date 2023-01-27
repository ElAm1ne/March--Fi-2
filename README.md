Projet de <b>sureplication (ou Super Hedge en anglais) d'options européennes</b>, plus exactement le <b>pricing</b> d'option européenne sous condition de suréplication parfaite sur toute la durée du <b>Backtest</b> (2000 à 2023).

Le but du projet est de créer et implémenter un modèle basé sur la <b>théorie des supports de prix</b>, ci-après le lien de l'article de recherche : https://hal.science/hal-02379707/document.

L'idée est de se baser sur l'historique des prix open, close, high, low afin de prédire les fluctuation journalière, <b>optimiser/calibrer</b> les paramètres du modèles sous deux contraintes :
- Maximiser le % de suréplication (jours surepliqués/n jours total) * (la différence entre la valeur finale du portefeuille et le payoff)
- Minimiser le prix de vente de l'option (V0 : valeur initiale du portefeuille).

Le <b>Colab</b> soumis à la fin du projet est le suivante : <b>https://github.com/ElAm1ne/March--Fi-2/blob/master/SuperHedge.ipynb</b>

Le choix d'utilisation de <b>Python + La librairie Numba</b> permet des temps de calculs proche du langage C/C++ tout en gardant la lisibilité du code Python.

Bonne lecture !
