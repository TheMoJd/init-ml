# init-ml

Small demo that trains a linear regression on `age_vs_poids_vs_taille_vs_sexe.csv`.

## Installation et exécution

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python init.py
```

## Interprétation des résultats

Le modèle de régression linéaire prédit le **poids** en fonction de trois variables : **sexe**, **âge**, et **taille**.

### Équation du modèle
Le modèle apprend l'équation suivante :
```
poids = 0.153×sexe + 0.108×age + 0.554×taille + intercept
```

### Coefficients
- **sexe (0.153)** : En moyenne, pour une même taille et un même âge, changer de sexe augmente le poids prédit d'environ 0.15 kg (si sexe est encodé 0/1).
- **age (0.108)** : Chaque année supplémentaire ajoute environ 0.11 kg au poids prédit, toutes choses égales par ailleurs.
- **taille (0.554)** : Chaque cm de taille supplémentaire augmente le poids prédit d'environ 0.55 kg. C'est le facteur le plus influent.

### Métriques de performance
- **R² (0.63)** : Le modèle explique environ 63% de la variance du poids. C'est un score correct mais il reste 37% de variabilité non expliquée par ces trois variables.
- **MSE (28.6 kg²)** : Erreur quadratique moyenne. La racine carrée (RMSE ≈ 5.3 kg) indique l'écart-type des erreurs.
- **MAE (4.16 kg)** : En moyenne, les prédictions s'écartent de ±4.2 kg du poids réel.
- **MAPE (9.05%)** : L'erreur moyenne en pourcentage est d'environ 9%, ce qui signifie que les prédictions sont généralement proches des valeurs réelles.

### Exemple de prédiction
Pour une personne avec `sexe=0`, `age=150` et `taille=153` cm, le modèle prédit un poids de **42.7 kg**.

### Conclusion
Le modèle montre que la **taille** est le prédicteur le plus important du poids, suivi par le sexe et l'âge. Les performances sont acceptables (R²=0.63, MAPE=9%) mais pourraient être améliorées en ajoutant d'autres variables ou en utilisant des modèles non-linéaires.
