import pandas as pd
import numpy as np

# lecture du jeu de données
df = pd.read_csv('datasets/age_vs_poids_vs_taille_vs_sexe.csv')

# Les variables prédictives
X = df[['sexe', 'age', 'taille',]]

# Variable cible: poids
y = df['poids']

# on choisit un modèle de régression linéaire
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# On entraine le modèle avec la méthode fit
reg.fit(X, y)

# et on obtient un score
print('R^2 score:', reg.score(X, y))

# ainsi que les coefficients a,b,c de la régression linéaire
print('Coefficients:', reg.coef_)

# Prédiction avec un DataFrame pour éviter le warning sklearn
sample = pd.DataFrame([[0, 150, 153]], columns=['sexe', 'age', 'taille'])
print('Prediction:', reg.predict(sample))

# Calculer les métriques d'erreur
y_pred = reg.predict(X)
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
print('MSE:', mean_squared_error(y, y_pred))
print('MAE:', mean_absolute_error(y, y_pred))
print('MAPE:', mean_absolute_percentage_error(y, y_pred))