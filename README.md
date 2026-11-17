# Edge_AI

Projet d'apprentissage de l'**Edge AI** : entraîner des modèles de machine learning sur PC, puis les exécuter directement sur des microcontrôleurs (ESP32), sans serveur ni connexion Internet.

## Intérêt

Faire tourner l'IA « au bord » (sur l'objet lui-même) plutôt que dans le cloud permet :

- **Latence faible** : la décision est prise localement, en quelques millisecondes.
- **Confidentialité** : les données (ex. médicales) ne quittent pas l'appareil.
- **Autonomie et faible consommation** : pas de réseau nécessaire, matériel bon marché.

La contrainte principale est la taille : un ESP32 n'a que ~320 Ko de RAM. Le projet explore donc comment obtenir des modèles **précis mais légers** (choix des variables, profondeur et nombre d'arbres, quantification).

## Démarche

1. **Entraînement en Python** (`Mo-dev/Python/`) avec scikit-learn : Random Forest, KNN, régression… sur plusieurs jeux de données (maladie cardiaque, cancer du sein, MNIST, etc. dans `Mo-dev/data/`).
2. **Analyse** : scores train/test, importance des variables, réduction du modèle.
3. **Export pour l'embarqué** : conversion du modèle en code C avec [emlearn](https://github.com/emlearn/emlearn) et sauvegarde des paramètres de normalisation.
4. **Déploiement** sur ESP32 (PlatformIO / Arduino) pour faire l'inférence sur la carte.
